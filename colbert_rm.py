"""
CachedColBERTRM — a drop-in replacement for lotus ColBERTv2RM that:
  - Saves docs and index to a single user-specified directory (no hidden
    experiments/lotus/indexes/… nesting).
  - Shows progress during indexing.
  - Caches the Searcher so it is not re-instantiated on every query.
"""

import pickle
import threading
import time
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm import tqdm

from lotus.models.rm import RM
from lotus.types import RMOutput
from lotus.vector_store.vs import VS

try:
    from colbert import Indexer, Searcher
    from colbert.infra import ColBERTConfig, Run, RunConfig
except ImportError as e:
    raise ImportError("Install colbert-ai: pip install colbert-ai") from e


_CHECKPOINT   = "colbert-ir/colbertv2.0"
_INDEX_NAME   = "index"        # sub-dir ColBERT writes inside the experiment root
_DOCS_FILE    = "docs.pkl"     # stored directly in index_dir


class CachedColBERTRM(RM):
    """
    ColBERTv2 retrieval model with persistent index caching and progress output.

    Usage
    -----
    rm = CachedColBERTRM()

    # First run — builds and saves the index:
    rm.index(docs, "my_index_dir")

    # Subsequent runs — loads from disk instantly:
    rm.load_index("my_index_dir")

    # Search (Searcher is cached after the first call):
    output = rm(["my query"], K=5)
    """

    def __init__(self, doc_maxlen: int = 300, nbits: int = 2) -> None:
        super().__init__()
        self.doc_maxlen  = doc_maxlen
        self.nbits       = nbits
        self.docs:       list[str] | None = None
        self.index_dir:  str | None       = None
        self._searcher:  Any | None       = None   # cached Searcher

    def _embed(self, docs: list[str]) -> NDArray[np.float64]:
        """Not used for indexing — ColBERTVS.index() handles that. Returns a dummy array."""
        return np.zeros((len(docs), 1), dtype=np.float64)

    def convert_query_to_query_vector(
        self,
        queries: Union["pd.Series", str, list[str], NDArray[np.float64]],
    ) -> list[str]:
        """Pass query text through unchanged so ColBERTVS can hand it to the searcher."""
        if isinstance(queries, str):
            return [queries]
        if isinstance(queries, pd.Series):
            return queries.tolist()
        if isinstance(queries, np.ndarray):
            raise TypeError("ColBERT requires text queries, not pre-computed vectors.")
        return list(queries)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _docs_path(self, index_dir: str) -> Path:
        return Path(index_dir) / _DOCS_FILE

    def _colbert_index_path(self, index_dir: str) -> str:
        """
        ColBERT's Indexer writes to  <root>/indexes/<name>/
        We set root=index_dir and name=_INDEX_NAME so everything lands
        under  <index_dir>/indexes/index/
        """
        return _INDEX_NAME

    def _run_config(self, index_dir: str) -> RunConfig:
        return RunConfig(nranks=1, experiment=_INDEX_NAME, root=str(Path(index_dir)))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index(self, docs: list[str], index_dir: str, **kwargs: Any) -> None:
        """
        Build a ColBERT index over docs and save everything to index_dir.
        Shows a preparation progress bar and elapsed time around the encode step.
        """
        Path(index_dir).mkdir(parents=True, exist_ok=True)

        # Show progress while validating / deduplicating the doc list
        print(f"[ColBERT] Preparing {len(docs):,} documents …", flush=True)
        validated: list[str] = []
        for doc in tqdm(docs, desc="  preparing docs", unit="doc", dynamic_ncols=True):
            validated.append(doc if doc.strip() else " ")   # ColBERT dislikes empty strings

        config = ColBERTConfig(
            doc_maxlen=kwargs.get("doc_maxlen", self.doc_maxlen),
            nbits=kwargs.get("nbits", self.nbits),
            kmeans_niters=kwargs.get("kmeans_niters", 4),
        )

        print(f"[ColBERT] Encoding & indexing → '{index_dir}' ")
        t0 = time.perf_counter()
        with Run().context(self._run_config(index_dir)):
            indexer = Indexer(checkpoint=_CHECKPOINT, config=config)
            indexer.index(name=_INDEX_NAME, collection=validated, overwrite=True)
        elapsed = time.perf_counter() - t0
        print(f"[ColBERT] Encoding done in {elapsed:.1f}s.", flush=True)

        # Pickle docs so load_index can restore them without re-encoding
        docs_path = self._docs_path(index_dir)
        print(f"[ColBERT] Saving {len(validated):,} docs to '{docs_path}' …", flush=True)
        with open(docs_path, "wb") as fp:
            pickle.dump(validated, fp, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[ColBERT] Index ready at '{index_dir}'.", flush=True)

        self.docs      = validated
        self.index_dir = index_dir
        self._searcher = None

    def load_index(self, index_dir: str) -> None:
        """Load a previously built index from disk."""
        docs_path = self._docs_path(index_dir)
        if not docs_path.exists():
            raise FileNotFoundError(
                f"Docs file not found at '{docs_path}'. "
                "Has the index been built with CachedColBERTRM.index()?"
            )
        print(f"[ColBERT] Loading docs from '{docs_path}' …", flush=True)
        t0 = time.perf_counter()
        with open(docs_path, "rb") as fp:
            self.docs = pickle.load(fp)
        elapsed = time.perf_counter() - t0
        print(f"[ColBERT] Loaded {len(self.docs):,} docs in {elapsed:.1f}s. "
              f"Searcher will be initialised on first query.", flush=True)

    def _get_searcher(self) -> "Searcher":
        """Return the cached Searcher, initialising it if needed."""
        if self._searcher is None:
            if self.index_dir is None:
                raise ValueError("No index loaded. Call index() or load_index() first.")
            print("[ColBERT] Initialising Searcher …", flush=True)
            with Run().context(self._run_config(self.index_dir)):
                self._searcher = Searcher(
                    index=_INDEX_NAME,
                    collection=self.docs,
                )
            print("[ColBERT] Searcher ready.", flush=True)
        return self._searcher

    def get_vectors_from_index(self, index_dir: str, ids: list[int]) -> NDArray[np.float64]:
        raise NotImplementedError("ColBERTv2 does not expose raw document vectors.")

    def __call__(
        self,
        queries: str | list[str] | NDArray[np.float64],
        K: int | None = None,
        **kwargs: Any,
    ) -> "RMOutput | NDArray[np.float64]":
        if K is None:
            # Called by sem_index for embedding — return dummy; ColBERTVS.index() does the real work
            if isinstance(queries, str):
                queries = [queries]
            return np.zeros((len(queries), 1), dtype=np.float64)
        if isinstance(queries, str):
            queries = [queries]
        assert isinstance(queries, list), "queries must be a str or list[str]"

        searcher    = self._get_searcher()
        queries_dict = {i: q for i, q in enumerate(queries)}
        all_results  = searcher.search_all(queries_dict, k=K).todict()

        indices   = [[r[0] for r in all_results[qid]] for qid in all_results]
        distances = [[r[2] for r in all_results[qid]] for qid in all_results]

        return RMOutput(distances=distances, indices=indices)


class ColBERTVS(VS):
    """
    A LOTUS VS adapter that routes sem_index / sem_search through CachedColBERTRM.

    When LOTUS calls vs.index(), we build the ColBERT index from the docs it passes.
    When LOTUS calls vs(queries, K), we hand the query text straight to the ColBERT searcher.

    Usage
    -----
    rm = CachedColBERTRM()
    vs = ColBERTVS(rm)
    lotus.settings.configure(lm=lm, rm=rm, vs=vs)

    # sem_index / load_sem_index then work transparently:
    df.sem_index("col", "my_index_dir")
    df.sem_search("col", "query", K=5)
    """

    def __init__(self, rm: CachedColBERTRM) -> None:
        super().__init__()
        self._rm = rm

    def index(self, docs: Any, embeddings: NDArray[np.float64], index_dir: str, **kwargs: Any) -> None:
        """Build the ColBERT index. `docs` is the pandas Series LOTUS passes; embeddings are ignored."""
        doc_list = docs.tolist() if hasattr(docs, "tolist") else list(docs)
        self._rm.index(doc_list, index_dir)
        self.index_dir = index_dir

    def load_index(self, index_dir: str) -> None:
        self._rm.load_index(index_dir)
        self.index_dir = index_dir

    def get_vectors_from_index(self, index_dir: str, ids: list[int]) -> NDArray[np.float64]:
        raise NotImplementedError("ColBERT does not expose raw document vectors.")

    def __call__(
        self,
        queries: Any,
        K: int,
        ids: list[int] | None = None,
        **kwargs: Any,
    ) -> RMOutput:
        """Search via ColBERT. `queries` is a list of strings from convert_query_to_query_vector."""
        if isinstance(queries, str):
            queries = [queries]

        rm_output = self._rm(queries, K if ids is None else max(K, len(ids)))

        if ids is None:
            return rm_output

        # Filter results to the requested id subset (best-effort approximation)
        ids_set = set(ids)
        filtered_indices: list[list[int]] = []
        filtered_distances: list[list[float]] = []
        for q_indices, q_distances in zip(rm_output.indices, rm_output.distances):
            pairs = [(idx, dist) for idx, dist in zip(q_indices, q_distances) if idx in ids_set][:K]
            filtered_indices.append([p[0] for p in pairs])
            filtered_distances.append([p[1] for p in pairs])
        return RMOutput(distances=filtered_distances, indices=filtered_indices)
