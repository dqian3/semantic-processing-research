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
    python experiment_biodex.py [--limit 250] [--model MODEL] [--provider anthropic|ollama]
                                [--top-k 10] [--approach agentic|lotus|both]
                                [--index-dir biodex_colbert_index]
                                [--output results.jsonl]

Examples:
    # Anthropic (default)
    python experiment_biodex.py --provider anthropic --model claude-haiku-4-5-20251001

    # Ollama (local llama3 8b — must have `ollama serve` running)
    python experiment_biodex.py --provider ollama --model llama3
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from pydantic import BaseModel

import litellm
import lotus
import pandas as pd
from datasets import load_dataset
from lotus.models import LM, SentenceTransformersRM
from lotus.types import CascadeArgs, UsageLimit
from lotus.vector_store import FaissVS
from colbert_rm import CachedColBERTRM, ColBERTVS

# litellm._turn_on_debug()

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

class ReactionList(BaseModel):
    reactions: list[str]


AGENTIC_SYSTEM = """\
You are a biomedical named-entity extractor. Given a patient medical article,
use the search_reactions tool to retrieve MedDRA drug reaction labels from the
catalogue, then select and rank the labels that match reactions described in
the article.

Rules:
- You MUST call search_reactions at least once before finishing.
- Do NOT repeat a query you have already searched — use different terms each time.
- Only include exact label strings returned by the search_reactions tool.
- Do not invent, paraphrase, or add any labels not returned by the tool.
- Do not include commentary, explanations, or non-label text."""

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
    full_model: str,
    article: str,
    labels_df: pd.DataFrame,
    max_turns: int = 5,
    search_k: int = 20,
    verbose: bool = False,
) -> list[str]:
    messages = [
        {"role": "system", "content": AGENTIC_SYSTEM},
        {"role": "user",   "content": f"ARTICLE:\n{article}"},
    ]

    for _ in range(max_turns):
        response = litellm.completion(
            model=full_model,
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
                if verbose:
                    print(f"  [tool] search_reactions({query!r}) → {retrieved['reaction'].tolist()[:5]} …")
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      results or "No matching reactions found.",
                })
        else:
            # Model finished searching — collect structured answer
            messages.append({"role": "user", "content": (
                "Output your final ranked reaction list. "
                "Use only exact label strings returned by the search_reactions tool — "
                "no paraphrasing, no invented terms, no commentary."
            )})
            final = litellm.completion(
                model=full_model,
                max_tokens=512,
                response_format=ReactionList,
                messages=messages,
            )
            content = final.choices[0].message.content or ""
            if verbose:
                print(f"  [agent] structured answer: {content[:200]}")
            try:
                return ReactionList.model_validate_json(content).reactions
            except Exception:
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


def lotus_predict(
    articles_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    top_k: int,
    verbose: bool = False,
) -> list[list[str]]:
    matched = articles_df[["id", "text"]].sem_join(
        labels_df,
        JOIN_PROMPT,
        cascade_args=CascadeArgs(recall_target=0.9),
    )

    if verbose:
        print(f"  [lotus] sem_join matched {len(matched)} (article, reaction) pairs across {len(articles_df)} articles")

    if matched.empty:
        return [[] for _ in range(len(articles_df))]

    # Rank reactions per article
    preds: dict[int, list[str]] = {}
    for article_id, group in matched.groupby("id"):
        ranked = group.sem_topk(RANK_PROMPT, K=top_k)
        if verbose:
            print(f"  [lotus] article {article_id}: {len(group)} candidates → top {top_k}: {ranked['reaction'].tolist()}")
        preds[article_id] = ranked["reaction"].tolist()

    return [preds.get(row.id, []) for row in articles_df.itertuples()]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BioDEX experiment: Agentic vs LOTUS")
    p.add_argument("--limit",     type=int, default=250)
    p.add_argument("--model",     default="claude-haiku-4-5-20251001",
                   help="Model name (e.g. claude-haiku-4-5-20251001 or llama3)")
    p.add_argument("--provider",  choices=["anthropic", "ollama"], default="anthropic",
                   help="Inference provider (default: anthropic). Use 'ollama' for local models.")
    p.add_argument("--top-k",     type=int, default=10, help="Max predictions per article")
    p.add_argument("--approach",  choices=["agentic", "lotus", "both"], default="both")
    p.add_argument("--index-dir", default="biodex_colbert_index",
                   help="Directory to cache ColBERT index over reaction labels")
    p.add_argument("--colbert", action="store_true",
                   help="Use ColBERT retriever (GPU required). Default: SentenceTransformers (CPU-friendly).")
    p.add_argument("--rate-limit", type=int, default=10,
                   help="Max LM requests per minute sent to the API (default: 50). Lower to avoid 429s.")
    p.add_argument("--max-budget", type=float, default=None,
                   help="Hard cost cap in USD for LOTUS LM calls (e.g. 1.0). No limit by default.")
    p.add_argument("--verbose", action="store_true",
                   help="Print tool calls and intermediate reasoning for each article.")
    p.add_argument("--output",    default=None, help="Write per-article results as JSONL")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Error: ANTHROPIC_API_KEY environment variable not set.")

    full_model = f"{args.provider}/{args.model}"
    print(f"Using model: {full_model}", flush=True)

    # Retry on 429 / transient errors for agentic litellm calls
    litellm.num_retries = 5

    usage_limit = (
        UsageLimit(total_cost_limit=args.max_budget)
        if args.max_budget is not None
        else UsageLimit()
    )
    lm = LM(
        model=full_model,
        rate_limit=args.rate_limit,
        physical_usage_limit=usage_limit,
    )
    if args.colbert:
        rm = CachedColBERTRM()
        vs = ColBERTVS(rm)
        lotus.settings.configure(lm=lm, rm=rm, vs=vs)
    else:
        rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
        lotus.settings.configure(lm=lm, rm=rm, vs=FaissVS())

    articles_df, labels_df = load_biodex(args.limit)
    index_ready = (
        (Path(args.index_dir) / "docs.pkl").exists()  # ColBERT
        if args.colbert
        else (Path(args.index_dir) / "index").exists()  # FaissVS
    )
    if index_ready:
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
            if args.verbose:
                print(f"\n[agentic] article {i}:")
            pred = agentic_predict(full_model, article_row.text, labels_df,
                                   search_k=args.top_k * 2, verbose=args.verbose)
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
        lotus_preds = lotus_predict(articles_df, labels_df, top_k=args.top_k, verbose=args.verbose)
        for i, (pred, article_row) in enumerate(zip(lotus_preds, articles_df.itertuples()), 1):
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
