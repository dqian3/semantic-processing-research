"""
Experiment: compare two fact-checking approaches on FEVER (sample.jsonl).

  1. Agentic   - LLM with a search_wikipedia tool; decides what to query.
  2. LOTUS     - map → search → filter pipeline over the local wiki corpus.

Labels are binarised following the LOTUS paper:
  SUPPORTS  → "supported"
  REFUTES / NOT ENOUGH INFO → "not supported"

Usage:
    python experiment.py [--sample sample.jsonl] [--wiki-dir wiki-pages]
                         [--limit N] [--model MODEL] [--provider anthropic|ollama]
                         [--approach agentic|lotus|both]
                         [--output results.jsonl]

Examples:
    # Anthropic (default)
    python experiment.py --provider anthropic --model claude-haiku-4-5-20251001

    # Ollama (local llama3 8b — must have `ollama serve` running)
    python experiment.py --provider ollama --model llama3
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import litellm
import lotus
import numpy as np
import pandas as pd
from lotus.dtype_extensions import convert_to_base_data
from lotus.models import LM, SentenceTransformersRM
from lotus.types import RMOutput
from lotus.vector_store import FaissVS
from tqdm import tqdm
from colbert_rm import CachedColBERTRM, ColBERTVS


# ---------------------------------------------------------------------------
# Memory-efficient RM + VS: stream embeddings to disk instead of np.vstack
# ---------------------------------------------------------------------------

class MemmapRM(SentenceTransformersRM):
    """Writes each embedding batch to a memmap file to avoid the np.vstack RAM spike."""

    _TMP = "./lotus_embed_tmp.mmap"

    def _embed(self, docs: list[str]) -> np.ndarray:
        dim = self.transformer.get_sentence_embedding_dimension()
        n = len(docs)
        mmap = np.memmap(self._TMP, dtype=np.float32, mode="w+", shape=(n, dim))
        for i in tqdm(range(0, n, self.max_batch_size), desc="Embedding"):
            batch = docs[i : i + self.max_batch_size]
            embs = self.transformer.encode(
                convert_to_base_data(batch),
                convert_to_tensor=True,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=False,
            )
            mmap[i : i + len(batch)] = embs.cpu().numpy()
            del embs
        mmap.flush()
        return mmap


class MemmapFaissVS(FaissVS):
    """Stores vecs as a float32 memmap on disk.

    Search: chunked numpy dot-product — no FAISS index loaded into RAM.
      Peak RAM = one chunk (~150 MB) regardless of corpus size.
      Speed: ~100–300 ms/query on CPU (fine when LLM calls dominate).

    Optional FAISS build (call build_faiss_from_vecs() + set use_faiss=True):
      IVF4096,SQ8 keeps the index at ~1.9 GB instead of 7.5 GB for Flat.
    """

    _FACTORY = "IVF4096,SQ8"
    _CHUNK   = 100_000
    _N_TRAIN = 200_000  # ≥ 39 × nlist recommended for IVF4096

    def __init__(self, factory_string: str | None = None, use_faiss: bool = False, **kwargs):
        super().__init__(factory_string=factory_string or self._FACTORY, **kwargs)
        self.use_faiss = use_faiss

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _shape_path(self, index_dir: str) -> str:
        return f"{index_dir}/vecs.shape"

    def _vecs_path(self, index_dir: str) -> str:
        return f"{index_dir}/vecs.f32"

    def _open_vecs(self, index_dir: str) -> np.ndarray:
        with open(self._shape_path(index_dir)) as f:
            n, dim = map(int, f.read().split(","))
        return np.memmap(self._vecs_path(index_dir), dtype=np.float32, mode="r", shape=(n, dim))

    # ------------------------------------------------------------------
    # Optional: build a FAISS IVF+SQ8 index for faster search
    # ------------------------------------------------------------------

    def build_faiss_from_vecs(self, index_dir: str) -> None:
        """Train an IVF+SQ8 FAISS index from existing vecs.f32 (skips re-encoding).

        After this succeeds, pass use_faiss=True to MemmapFaissVS to use it.
        """
        import faiss as _faiss
        print(f"Building IVF+SQ8 FAISS index from vecs in '{index_dir}' …", flush=True)
        src = self._open_vecs(index_dir)
        n, dim = src.shape

        index = _faiss.index_factory(dim, self.factory_string, self.metric)

        n_train = min(n, self._N_TRAIN)
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(n, n_train, replace=False))
        train_vecs = np.ascontiguousarray(src[idx], dtype=np.float32)
        print(f"  Training on {n_train:,} samples …", flush=True)
        index.train(train_vecs)
        del train_vecs

        for i in tqdm(range(0, n, self._CHUNK), desc="Adding to FAISS"):
            index.add(np.ascontiguousarray(src[i : i + self._CHUNK], dtype=np.float32))

        _faiss.write_index(index, f"{index_dir}/index")
        self.faiss_index = index
        self.index_dir = index_dir
        print("FAISS index written. Re-run with use_faiss=True to use it.", flush=True)

    # ------------------------------------------------------------------
    # Required VS interface
    # ------------------------------------------------------------------

    def index(self, docs: list[str], embeddings: np.ndarray, index_dir: str, **kwargs) -> None:
        n, dim = embeddings.shape
        os.makedirs(index_dir, exist_ok=True)
        with open(self._shape_path(index_dir), "w") as f:
            f.write(f"{n},{dim}")
        out = np.memmap(self._vecs_path(index_dir), dtype=np.float32, mode="w+", shape=(n, dim))
        for i in range(0, n, self._CHUNK):
            out[i : i + self._CHUNK] = embeddings[i : i + self._CHUNK]
        out.flush()
        del out
        Path(f"{index_dir}/index").touch()  # sentinel for build_or_load_index
        self.index_dir = index_dir

    def load_index(self, index_dir: str) -> None:
        self.index_dir = index_dir
        self.vecs = self._open_vecs(index_dir)
        if self.use_faiss:
            import faiss as _faiss
            self.faiss_index = _faiss.read_index(f"{index_dir}/index")

    def get_vectors_from_index(self, index_dir: str, ids: list[int]) -> np.ndarray:
        return self._open_vecs(index_dir)[ids]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def __call__(self, query_vectors: np.ndarray, K: int, ids: list[int] | None = None, **kwargs) -> RMOutput:
        if self.use_faiss:
            return super().__call__(query_vectors, K, ids=ids, **kwargs)

        q = np.ascontiguousarray(query_vectors, dtype=np.float32)
        vecs = self._open_vecs(self.index_dir)

        if ids is not None:
            subset = np.ascontiguousarray(vecs[ids], dtype=np.float32)
            scores = q @ subset.T
            local_k = min(K, len(subset))
            top = np.argsort(-scores, axis=1)[:, :local_k]
            ids_arr = np.array(ids)
            return RMOutput(
                distances=np.take_along_axis(scores, top, axis=1).tolist(),
                indices=ids_arr[top].tolist(),
            )

        n = len(vecs)
        chunk_scores, chunk_indices = [], []
        for i in range(0, n, self._CHUNK):
            chunk = np.ascontiguousarray(vecs[i : i + self._CHUNK], dtype=np.float32)
            cs = q @ chunk.T
            local_k = min(K, cs.shape[1])
            top = np.argpartition(-cs, local_k - 1, axis=1)[:, :local_k]
            chunk_scores.append(np.take_along_axis(cs, top, axis=1))
            chunk_indices.append(top + i)

        all_scores = np.concatenate(chunk_scores, axis=1)
        all_indices = np.concatenate(chunk_indices, axis=1)
        top_k = np.argpartition(-all_scores, K - 1, axis=1)[:, :K]
        final_scores = np.take_along_axis(all_scores, top_k, axis=1)
        final_indices = np.take_along_axis(all_indices, top_k, axis=1)
        order = np.argsort(-final_scores, axis=1)
        return RMOutput(
            distances=np.take_along_axis(final_scores, order, axis=1).tolist(),
            indices=np.take_along_axis(final_indices, order, axis=1).tolist(),
        )

# ---------------------------------------------------------------------------
# Labels (binary, following LOTUS paper)
# ---------------------------------------------------------------------------

SUPPORTED     = "supported"
NOT_SUPPORTED = "not supported"
LABELS        = [SUPPORTED, NOT_SUPPORTED]

def gold_label(raw: str) -> str:
    """Map FEVER 3-way label to binary."""
    return SUPPORTED if raw.upper() == "SUPPORTS" else NOT_SUPPORTED


# ---------------------------------------------------------------------------
# Token / cost tracking via litellm.success_callback
# Fires on every litellm.completion call — covers both agentic and LOTUS.
# ---------------------------------------------------------------------------

class _UsageAccumulator:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.input_tokens  = 0
        self.output_tokens = 0
        self.cost          = 0.0

    def __call__(self, kwargs, completion_response, start_time, end_time) -> None:
        usage = getattr(completion_response, "usage", None)
        if usage:
            self.input_tokens  += getattr(usage, "prompt_tokens",     0)
            self.output_tokens += getattr(usage, "completion_tokens", 0)
        self.cost += completion_response._hidden_params.get("response_cost", 0.0) or 0.0

tracker = _UsageAccumulator()
litellm.success_callback = [tracker]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_claims(path: str, limit: int | None = None) -> list[dict]:
    claims = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                claims.append(json.loads(line))
    return claims[:limit] if limit else claims


def load_wiki_df(wiki_dir: str) -> pd.DataFrame:
    """Load all wiki sentences into a DataFrame with columns [title, text]."""
    rows = []
    for f in sorted(Path(wiki_dir).glob("wiki-*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                page = json.loads(raw)
                title = page.get("id", "")
                if not title:
                    continue
                for line in page.get("lines", "").split("\n"):
                    parts = line.split("\t")
                    if len(parts) >= 2 and parts[0].isdigit():
                        text = parts[1].strip()
                        if text:
                            rows.append({"title": title, "text": text})
    print(f"Loaded {len(rows):,} wiki sentences.", flush=True)
    return pd.DataFrame(rows)


def build_or_load_index(df: pd.DataFrame, col: str, index_dir: str, colbert: bool = False) -> pd.DataFrame:
    """Build an index on first run, load it on subsequent runs."""
    index_ready = (
        (Path(index_dir) / "docs.pkl").exists()  # ColBERT
        if colbert
        else (Path(index_dir) / "index").exists()  # FaissVS
    )
    vecs_only = (
        not colbert
        and not index_ready
        and (Path(index_dir) / "vecs.f32").exists()
        and (Path(index_dir) / "vecs.shape").exists()
    )
    if index_ready:
        print(f"Loading cached index from '{index_dir}' …", flush=True)
        df.load_sem_index(col, index_dir)
    elif vecs_only:
        # Embeddings already written — just create the sentinel and load
        print(f"Found existing vecs in '{index_dir}', skipping re-encoding …", flush=True)
        Path(index_dir, "index").touch()
        df.load_sem_index(col, index_dir)
    else:
        print(f"Building index → '{index_dir}' (one-time) …", flush=True)
        df.sem_index(col, index_dir)
    return df


# ---------------------------------------------------------------------------
# Approach 1: Agentic (tool-augmented, searches same wiki corpus)
# ---------------------------------------------------------------------------

AGENTIC_SYSTEM = """\
You are a fact-checking assistant. Given a CLAIM, use the search_wikipedia tool
to retrieve evidence from the Wikipedia corpus, then decide whether the claim is
supported or not supported.

When you have enough evidence, output ONLY one of:
  supported
  not supported"""

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_wikipedia",
        "description": "Search the Wikipedia corpus for sentences relevant to a query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"],
        },
    },
}


def agentic_classify(
    full_model: str,
    claim: str,
    wiki_df: pd.DataFrame,
    max_turns: int = 4,
) -> str:
    messages = [
        {"role": "system",  "content": AGENTIC_SYSTEM},
        {"role": "user",    "content": f"CLAIM: {claim}"},
    ]

    for _ in range(max_turns):
        response = litellm.completion(
            model=full_model,
            max_tokens=256,
            tools=[SEARCH_TOOL],
            messages=messages,
        )
        msg = response.choices[0].message
        messages.append(msg)

        if msg.tool_calls:
            for tc in msg.tool_calls:
                query     = json.loads(tc.function.arguments).get("query", "")
                retrieved = wiki_df.sem_search("text", query, K=TOP_K)
                passages  = "\n".join(
                    f"[{r.title}] {r.text}" for r in retrieved.itertuples()
                )
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      passages or "No results found.",
                })
        else:
            raw = (msg.content or "").strip().lower()
            return SUPPORTED if raw.startswith("supported") else NOT_SUPPORTED

    return NOT_SUPPORTED


# ---------------------------------------------------------------------------
# Approach 2: LOTUS  map → search → filter
#
# Follows the paper's pipeline:
#   1. sem_map:    claim  → search query
#   2. sem_search: query  → top-k wiki sentences
#   3. sem_filter: (claim, evidence) → supported?
# ---------------------------------------------------------------------------

TOP_K = 5

QUERY_GEN_PROMPT = (
    "Generate a short search query to find Wikipedia evidence for this claim.\n"
    "Claim: {claim}\nQuery:"
)

FILTER_PROMPT = (
    "Does the following Wikipedia passage support the claim?\n"
    'Claim: "{claim}"\n'
    'Passage: "{text}"\n'
    "Answer yes only if the passage clearly supports the claim."
)


def lotus_classify(claim: str, wiki_df: pd.DataFrame) -> str:
    claim_df = pd.DataFrame([{"claim": claim}])

    # Step 1: map claim → search query
    claim_df = claim_df.sem_map(QUERY_GEN_PROMPT, output_cols=["query"])
    query = claim_df.iloc[0]["query"]

    # Step 2: search wiki for relevant sentences
    retrieved = wiki_df.sem_search("text", query, K=TOP_K)
    retrieved = retrieved.assign(claim=claim)

    # Step 3: filter — does any retrieved passage support the claim?
    supported_rows = retrieved.sem_filter(FILTER_PROMPT)
    return SUPPORTED if len(supported_rows) > 0 else NOT_SUPPORTED


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    name: str,
    predictions: list[str],
    gold: list[str],
    elapsed: float,
    input_tokens: int,
    output_tokens: int,
    cost: float,
) -> dict:
    total = len(gold)
    tp = sum(p == SUPPORTED     and g == SUPPORTED     for p, g in zip(predictions, gold))
    tn = sum(p == NOT_SUPPORTED and g == NOT_SUPPORTED for p, g in zip(predictions, gold))
    fp = sum(p == SUPPORTED     and g == NOT_SUPPORTED for p, g in zip(predictions, gold))
    fn = sum(p == NOT_SUPPORTED and g == SUPPORTED     for p, g in zip(predictions, gold))
    correct   = tp + tn
    accuracy  = correct / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(name=name, total=total, correct=correct, accuracy=accuracy,
                precision=precision, recall=recall, f1=f1,
                tp=tp, tn=tn, fp=fp, fn=fn,
                elapsed=elapsed, input_tokens=input_tokens,
                output_tokens=output_tokens, cost=cost)


def print_report(metrics: list[dict]) -> None:
    W = 78
    print(f"\n{'='*W}")
    print("Comparison")
    print(f"{'='*W}")
    header = (f"{'Method':<28} {'Acc':>5} {'Prec':>5} {'Rec':>5} {'F1':>5}"
              f"  {'ET(s)':>6}  {'In tok':>7}  {'Out tok':>7}  {'Cost $':>7}")
    print(header)
    print("-" * W)
    for m in metrics:
        print(
            f"{m['name']:<28} {m['accuracy']:>5.1%} {m['precision']:>5.1%}"
            f" {m['recall']:>5.1%} {m['f1']:>5.1%}"
            f"  {m['elapsed']:>6.1f}  {m['input_tokens']:>7,}  {m['output_tokens']:>7,}"
            f"  ${m['cost']:>6.4f}"
        )
    print()
    for m in metrics:
        print(f"[{m['name']}] confusion matrix (positive = supported):")
        print(f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fact-checking experiment: Agentic vs LOTUS")
    p.add_argument("--sample", default="FEVER/sample.jsonl")
    p.add_argument("--wiki-dir", default="FEVER/wiki-pages")
    p.add_argument("--index-dir", default="wiki_index",
                   help="Directory to cache the index (built on first run)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--model", default="claude-haiku-4-5-20251001",
                   help="Model name (e.g. claude-haiku-4-5-20251001 or llama3)")
    p.add_argument("--provider", choices=["anthropic", "ollama"], default="anthropic",
                   help="Inference provider (default: anthropic). Use 'ollama' for local models.")
    p.add_argument("--approach", choices=["agentic", "lotus", "both"], default="both")
    p.add_argument("--colbert", action="store_true",
                   help="Use ColBERT retriever (GPU required). Default: SentenceTransformers.")
    p.add_argument("--output", default=None, help="Write per-claim results as JSONL")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Error: ANTHROPIC_API_KEY environment variable not set.")

    full_model = f"{args.provider}/{args.model}"
    print(f"Using model: {full_model}", flush=True)

    lm = LM(model=full_model)
    if args.colbert:
        rm = CachedColBERTRM()
        vs = ColBERTVS(rm)
    else:
        rm = MemmapRM(model="intfloat/e5-small-v2", device="cuda")
        vs = MemmapFaissVS()
    lotus.settings.configure(lm=lm, rm=rm, vs=vs)

    claims = load_claims(args.sample, args.limit)
    run_agentic = args.approach in ("agentic", "both")
    run_lotus   = args.approach in ("lotus",   "both")

    wiki_df = load_wiki_df(args.wiki_dir)
    wiki_df = build_or_load_index(wiki_df, "text", args.index_dir, colbert=args.colbert)

    gold_labels:   list[str] = [gold_label(c["label"]) for c in claims]
    agentic_preds: list[str] = []
    lotus_preds:   list[str] = []
    agentic_time = lotus_time = 0.0
    # snapshot tracker state between approaches
    agentic_usage: dict = {}
    lotus_usage:   dict = {}
    rows: list[dict] = []

    print(f"\nEvaluating {len(claims)} claims …\n")

    # --- agentic pass ---
    if run_agentic:
        tracker.reset()
        t_start = time.perf_counter()
        for i, item in enumerate(claims, 1):
            claim = item["claim"]
            gold  = gold_label(item["label"])
            pred  = agentic_classify(full_model, claim, wiki_df)
            agentic_preds.append(pred)
            mark = "✓" if pred == gold else "✗"
            print(f"[agentic {i:>4}/{len(claims)}] {mark}  gold={gold:<14} pred={pred:<14}  {claim[:50]}")
        agentic_time  = time.perf_counter() - t_start
        agentic_usage = {"input": tracker.input_tokens, "output": tracker.output_tokens, "cost": tracker.cost}

    # --- lotus pass ---
    if run_lotus:
        tracker.reset()
        t_start = time.perf_counter()
        for i, item in enumerate(claims, 1):
            claim = item["claim"]
            gold  = gold_label(item["label"])
            pred  = lotus_classify(claim, wiki_df)
            lotus_preds.append(pred)
            mark = "✓" if pred == gold else "✗"
            print(f"[lotus   {i:>4}/{len(claims)}] {mark}  gold={gold:<14} pred={pred:<14}  {claim[:50]}")
        lotus_time  = time.perf_counter() - t_start
        lotus_usage = {"input": tracker.input_tokens, "output": tracker.output_tokens, "cost": tracker.cost}

    # merge per-claim rows for output
    for i, item in enumerate(claims):
        row: dict = {"id": item["id"], "claim": item["claim"], "gold": gold_labels[i]}
        if run_agentic:
            row["agentic"] = agentic_preds[i]
        if run_lotus:
            row["lotus"] = lotus_preds[i]
        rows.append(row)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    all_metrics = []
    if run_agentic:
        all_metrics.append(evaluate(
            "Agentic", agentic_preds, gold_labels, agentic_time,
            agentic_usage["input"], agentic_usage["output"], agentic_usage["cost"],
        ))
    if run_lotus:
        all_metrics.append(evaluate(
            "LOTUS map-search-filter", lotus_preds, gold_labels, lotus_time,
            lotus_usage["input"], lotus_usage["output"], lotus_usage["cost"],
        ))

    print_report(all_metrics)


if __name__ == "__main__":
    main()
