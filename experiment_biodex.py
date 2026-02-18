"""
Experiment: extreme multi-label drug-reaction classification on BioDEX.

Compares:
  1. Agentic - LLM with a search_reactions tool; returns a ranked label list.
  2. LOTUS   - sem_join (articles x reactions) + sem_topk listwise ranking.

Metric: Rank-Precision@K (RP@K), following prior work.
  RP@K = (# true reactions in top-K predictions) / K

Dataset: BioDEX from HuggingFace (BioDEX/BioDEX-Reactions).
  Sample size: 250 articles (following the paper).

Usage:
    python experiment_biodex.py [--limit 250] [--model MODEL] [--top-k 10]
                                [--approach agentic|lotus|both]
                                [--index-dir biodex_colbert_index]
                                [--output results.jsonl]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import litellm
import lotus
import pandas as pd
from datasets import load_dataset
from lotus.models import LM, SentenceTransformersRM
from lotus.vector_store import FaissVS
from colbert_rm import CachedColBERTRM

# ---------------------------------------------------------------------------
# Token / cost tracking — same callback pattern as experiment.py
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

def load_biodex(limit: int = 250) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      articles_df  - columns: [id, text, reactions]  (sampled test articles)
      labels_df    - columns: [reaction]              (ALL unique MedDRA labels across every split)

    Labels are collected from all splits so we have the full ~24K label space,
    not just the reactions present in the sampled articles.
    """
    print("Loading BioDEX dataset from HuggingFace …", flush=True)
    dataset = load_dataset("BioDEX/BioDEX-Reactions")

    # Collect all unique reaction labels from every split
    all_reactions: set[str] = set()
    for split in dataset:
        for item in dataset[split]:
            for r in item["reactions"].split(", "):
                r = r.lower().strip()
                if r:
                    all_reactions.add(r)

    # Sample evaluation articles from the test split
    test_ds = dataset["test"].select(range(min(limit, len(dataset["test"]))))
    rows = []
    for i, item in enumerate(test_ds):
        rows.append({
            "id":        i,
            "text":      item["fulltext_processed"],
            "reactions": [r.lower().strip() for r in item["reactions"].split(", ") if r.strip()],
        })

    articles_df = pd.DataFrame(rows)
    labels_df   = pd.DataFrame({"reaction": sorted(all_reactions)})
    print(f"Loaded {len(articles_df)} test articles, {len(labels_df):,} unique reaction labels.", flush=True)
    print(labels_df, flush=True)
    print(articles_df, flush=True)
    return articles_df, labels_df



# ---------------------------------------------------------------------------
# Metric: Rank-Precision@K
# ---------------------------------------------------------------------------

def rp_at_k(predicted: list[str], gold: list[str], k: int) -> float:
    """Rank-Precision@K: fraction of top-K predictions that are true labels."""
    top_k = predicted[:k]
    hits  = sum(1 for p in top_k if p.lower().strip() in {g.lower().strip() for g in gold})
    return hits / k if k else 0.0


def evaluate(
    name: str,
    all_preds: list[list[str]],
    all_gold: list[list[str]],
    ks: list[int],
    elapsed: float,
    input_tokens: int,
    output_tokens: int,
    cost: float,
) -> dict:
    n = len(all_gold)
    scores = {k: sum(rp_at_k(p, g, k) for p, g in zip(all_preds, all_gold)) / n for k in ks}
    return {"name": name, "n": n, "rp": scores,
            "elapsed": elapsed, "input_tokens": input_tokens,
            "output_tokens": output_tokens, "cost": cost}


def print_report(metrics: list[dict], ks: list[int]) -> None:
    W = 78
    print(f"\n{'='*W}")
    print("Comparison")
    print(f"{'='*W}")
    header = (f"{'Method':<28}" + "".join(f"  RP@{k:>2}" for k in ks)
              + f"  {'ET(s)':>6}  {'In tok':>7}  {'Out tok':>7}  {'Cost $':>7}")
    print(header)
    print("-" * W)
    for m in metrics:
        row = (f"{m['name']:<28}" + "".join(f"  {m['rp'][k]:.3f}" for k in ks)
               + f"  {m['elapsed']:>6.1f}  {m['input_tokens']:>7,}"
               + f"  {m['output_tokens']:>7,}  ${m['cost']:>6.4f}")
        print(row)
    print()


# ---------------------------------------------------------------------------
# Approach 1: Agentic (search_reactions tool → ranked label list)
# ---------------------------------------------------------------------------

AGENTIC_SYSTEM = """\
You are a biomedical expert. Given a patient medical article, use the
search_reactions tool to explore possible drug reaction labels, then output
a ranked list of the most likely drug reactions for the patient.

Output ONLY a JSON array of reaction label strings, ranked most-likely first.
Example: ["nausea", "vomiting", "headache"]"""

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_reactions",
        "description": (
            "Search the MedDRA drug-reaction label catalogue for terms relevant to a query. "
            "Returns the closest matching reaction labels."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Medical symptom or reaction query"}
            },
            "required": ["query"],
        },
    },
}


def agentic_predict(
    model: str,
    article: str,
    labels_df: pd.DataFrame,
    max_turns: int = 5,
    search_k: int = 20,
) -> list[str]:
    messages = [
        {"role": "system", "content": AGENTIC_SYSTEM},
        {"role": "user",   "content": f"ARTICLE:\n{article[:4000]}"},
    ]

    for _ in range(max_turns):
        response = litellm.completion(
            model=f"anthropic/{model}",
            max_tokens=512,
            tools=[SEARCH_TOOL],
            messages=messages,
        )
        msg = response.choices[0].message
        messages.append(msg)

        if msg.tool_calls:
            for tc in msg.tool_calls:
                query     = json.loads(tc.function.arguments).get("query", "")
                retrieved = labels_df.sem_search("reaction", query, K=search_k)
                results   = "\n".join(retrieved["reaction"].tolist())
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      results or "No matching reactions found.",
                })
        else:
            raw = (msg.content or "").strip()
            try:
                start = raw.index("[")
                end   = raw.rindex("]") + 1
                return json.loads(raw[start:end])
            except (ValueError, json.JSONDecodeError):
                return []

    return []


# ---------------------------------------------------------------------------
# Approach 2: LOTUS  sem_join → sem_topk listwise ranking
# ---------------------------------------------------------------------------

JOIN_PROMPT = (
    "Does the medical article describe a patient who experienced the drug reaction '{reaction}'?\n"
    "Article (excerpt): {text}\n"
    "Answer yes only if the reaction is clearly described."
)

RANK_PROMPT = (
    "Rank the following drug reactions from most to least likely for the patient "
    "described in the article.\nArticle: {text}\nReactions: {reaction}"
)


def lotus_predict(article: str, labels_df: pd.DataFrame, top_k: int) -> list[str]:
    article_df = pd.DataFrame([{"text": article[:4000]}])

    matched = article_df.sem_join(labels_df, JOIN_PROMPT)

    if matched.empty:
        return []

    # Step 2: listwise ranking via sem_topk
    ranked = matched.sem_topk(RANK_PROMPT, K=top_k)
    return ranked["reaction"].tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BioDEX experiment: Agentic vs LOTUS")
    p.add_argument("--limit",     type=int, default=250)
    p.add_argument("--model",     default="claude-haiku-4-5-20251001")
    p.add_argument("--top-k",     type=int, default=10, help="Max predictions per article")
    p.add_argument("--approach",  choices=["agentic", "lotus", "both"], default="both")
    p.add_argument("--index-dir", default="biodex_colbert_index",
                   help="Directory to cache ColBERT index over reaction labels")
    p.add_argument("--colbert", action="store_true",
                   help="Use ColBERT retriever (GPU required). Default: SentenceTransformers (CPU-friendly).")
    p.add_argument("--output",    default=None, help="Write per-article results as JSONL")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Error: ANTHROPIC_API_KEY environment variable not set.")

    lm = LM(model=f"anthropic/{args.model}")
    if args.colbert:
        rm = CachedColBERTRM()
        lotus.settings.configure(lm=lm, rm=rm)
    else:
        rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
        lotus.settings.configure(lm=lm, rm=rm, vs=FaissVS())

    articles_df, labels_df = load_biodex(args.limit)
    if Path(args.index_dir).exists():
        print(f"Loading cached index from '{args.index_dir}' …", flush=True)
        labels_df.load_sem_index("reaction", args.index_dir)
    else:
        print(f"Building index → '{args.index_dir}' (one-time) …", flush=True)
        labels_df.sem_index("reaction", args.index_dir)

    run_agentic = args.approach in ("agentic", "both")
    run_lotus   = args.approach in ("lotus",   "both")

    eval_ks      = [5, 10]
    gold_all     = articles_df["reactions"].tolist()
    agentic_preds: list[list[str]] = []
    lotus_preds:   list[list[str]] = []
    agentic_time = lotus_time = 0.0
    agentic_usage: dict = {}
    lotus_usage:   dict = {}

    print(f"\nEvaluating {len(articles_df)} articles …\n")

    # --- agentic pass ---
    if run_agentic:
        tracker.reset()
        t_start = time.perf_counter()
        for i, article_row in enumerate(articles_df.itertuples(), 1):
            pred = agentic_predict(args.model, article_row.text, labels_df,
                                   search_k=args.top_k * 2)
            agentic_preds.append(pred)
            rp5  = rp_at_k(pred, article_row.reactions, 5)
            rp10 = rp_at_k(pred, article_row.reactions, 10)
            print(f"[agentic {i:>4}/{len(articles_df)}]  RP@5={rp5:.2f}  RP@10={rp10:.2f}  gold={article_row.reactions[:3]}  pred={pred[:5]} …")
        agentic_time  = time.perf_counter() - t_start
        agentic_usage = {"input": tracker.input_tokens, "output": tracker.output_tokens, "cost": tracker.cost}

    # --- lotus pass ---
    if run_lotus:
        tracker.reset()
        t_start = time.perf_counter()
        for i, article_row in enumerate(articles_df.itertuples(), 1):
            pred = lotus_predict(article_row.text, labels_df, top_k=args.top_k)
            lotus_preds.append(pred)
            rp5  = rp_at_k(pred, article_row.reactions, 5)
            rp10 = rp_at_k(pred, article_row.reactions, 10)
            print(f"[lotus   {i:>4}/{len(articles_df)}]  RP@5={rp5:.2f}  RP@10={rp10:.2f}  gold={article_row.reactions[:3]}  pred={pred[:5]} …")
        lotus_time  = time.perf_counter() - t_start
        lotus_usage = {"input": tracker.input_tokens, "output": tracker.output_tokens, "cost": tracker.cost}

    if args.output:
        rows = []
        for i, article_row in enumerate(articles_df.itertuples()):
            row: dict = {"id": article_row.id, "gold": article_row.reactions}
            if run_agentic: row["agentic"] = agentic_preds[i]
            if run_lotus:   row["lotus"]   = lotus_preds[i]
            rows.append(row)
        with open(args.output, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    all_metrics = []
    if run_agentic:
        all_metrics.append(evaluate(
            "Agentic", agentic_preds, gold_all, eval_ks, agentic_time,
            agentic_usage["input"], agentic_usage["output"], agentic_usage["cost"],
        ))
    if run_lotus:
        all_metrics.append(evaluate(
            "LOTUS join+rank", lotus_preds, gold_all, eval_ks, lotus_time,
            lotus_usage["input"], lotus_usage["output"], lotus_usage["cost"],
        ))

    print_report(all_metrics, eval_ks)


if __name__ == "__main__":
    main()
