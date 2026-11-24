"""Free-text search index and scorer (T6.2, option A from T6.1).

Build (laptop, once per data change):

    STORAGE_BACKEND=local python -m movierec.index.search --run-date 2026-11-22

reads run D's TF-IDF vectorizer, matrix and row index and the catalog's `id`
column through get_store(), checks they describe the same films, and writes
three files next to each other at config `search.index_path` (a path stem):

    <stem>.joblib  the run's fitted TfidfVectorizer, byte-for-byte
    <stem>.npz     the run's TF-IDF matrix, byte-for-byte
    <stem>.json    run_date, movie id per matrix row, shapes, source checksums

No refitting and no extra fields (Phase 6 rules): the index is the run's own
representation. The catalog is read with `columns=CATALOG_COLUMNS` only; the
`keywords` column is the held-out label and is never read (CLAUDE.md).

Score: SearchIndex.query(text) vectorises the text with the run's own
vectorizer (its analyzer, vocabulary and IDF), takes cosine similarity
against every row, and returns the top `search.top_k` films with a positive
score, ties broken by movie id ascending. No reranking. An empty query, or
one with fewer than `search.min_query_terms` vocabulary terms, returns no
results and a reason, never arbitrary films.

pandas is imported only by the builder, so the API (T6.4) can import
SearchIndex with scikit-learn, scipy and numpy alone.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "default.yaml"

INDEX_PATH_ENV = "MOVIEREC_SEARCH_INDEX_PATH"
METHOD = "tfidf"
# The only catalog column the builder reads. Never add "keywords".
CATALOG_COLUMNS = ("id",)
ROW_INDEX_COLUMNS = ("row", "movie_id")
SUFFIXES = (".joblib", ".npz", ".json")

# Reasons a query returns no results.
REASON_EMPTY = "empty_query"
REASON_NO_TERMS = "no_known_terms"
REASON_TOO_FEW_TERMS = "too_few_known_terms"


def load_search_config() -> dict:
    """The `search` block of config/default.yaml (top_k, min_query_terms,
    index_path). Missing keys raise: thresholds live in config only."""
    with open(CONFIG_PATH) as f:
        search = yaml.safe_load(f)["search"]
    for key in ("top_k", "min_query_terms", "index_path"):
        if key not in search:
            raise KeyError(f"config search.{key} is missing")
    return search


def default_index_path() -> str:
    """Env MOVIEREC_SEARCH_INDEX_PATH, then config search.index_path."""
    return os.environ.get(INDEX_PATH_ENV) or load_search_config()["index_path"]


def index_files(stem: str | Path) -> dict[str, Path]:
    """{".joblib": path, ".npz": path, ".json": path} for an index stem."""
    stem = Path(stem)
    return {suffix: stem.with_name(stem.name + suffix) for suffix in SUFFIXES}


# --------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------


def build_search_index(store, run_date: str, out_stem: str | Path) -> dict:
    """Read run `run_date` from `store`, validate it, write the index files.

    Returns the metadata written to <stem>.json.
    """
    import joblib
    import pandas as pd
    import scipy.sparse as sp

    prefix = f"{METHOD}/{run_date}"
    vectorizer_bytes = store.get_bytes(f"{prefix}/vectorizer.joblib")
    matrix_bytes = store.get_bytes(f"{prefix}/matrix.npz")
    row_index = pd.read_parquet(
        io.BytesIO(store.get_bytes(f"{prefix}/row_index.parquet")),
        columns=list(ROW_INDEX_COLUMNS),
    )
    catalog = pd.read_parquet(
        io.BytesIO(store.get_bytes(f"catalog/{run_date}/catalog.parquet")),
        columns=list(CATALOG_COLUMNS),
    )

    vectorizer = joblib.load(io.BytesIO(vectorizer_bytes))
    matrix = sp.load_npz(io.BytesIO(matrix_bytes))

    row_index = row_index.sort_values("row")
    n_rows = len(row_index)
    if row_index["row"].tolist() != list(range(n_rows)):
        raise ValueError("row_index rows are not 0..n-1")
    movie_ids = [int(i) for i in row_index["movie_id"]]
    if len(set(movie_ids)) != n_rows:
        raise ValueError("row_index has duplicate movie ids")
    if matrix.shape[0] != n_rows:
        raise ValueError(f"matrix has {matrix.shape[0]} rows, row_index {n_rows}")
    n_features = len(vectorizer.vocabulary_)
    if matrix.shape[1] != n_features:
        raise ValueError(f"matrix has {matrix.shape[1]} columns, vocabulary {n_features}")
    catalog_ids = {int(i) for i in catalog["id"]}
    if catalog_ids != set(movie_ids) or len(catalog) != n_rows:
        raise ValueError("row_index movie ids do not match the catalog's ids")

    meta = {
        "run_date": run_date,
        "method": METHOD,
        "n_rows": n_rows,
        "n_features": n_features,
        "nnz": int(matrix.nnz),
        "vectorizer_sha256": hashlib.sha256(vectorizer_bytes).hexdigest(),
        "matrix_sha256": hashlib.sha256(matrix_bytes).hexdigest(),
        "movie_ids": movie_ids,
    }

    paths = index_files(out_stem)
    paths[".json"].parent.mkdir(parents=True, exist_ok=True)
    paths[".joblib"].write_bytes(vectorizer_bytes)
    paths[".npz"].write_bytes(matrix_bytes)
    paths[".json"].write_text(json.dumps(meta, separators=(",", ":")), encoding="utf-8")
    return meta


# --------------------------------------------------------------------------
# Scorer
# --------------------------------------------------------------------------


@dataclass
class SearchResult:
    """`ids` (str, as DynamoDB keys) and `scores` (cosine), best first.
    `reason` is None when the query was scored, else why it returned nothing."""

    ids: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    reason: str | None = None
    n_terms: int = 0


class SearchIndex:
    def __init__(self, vectorizer, matrix, movie_ids, run_date: str,
                 top_k: int, min_query_terms: int):
        import numpy as np
        from sklearn.preprocessing import normalize

        if matrix.shape[0] != len(movie_ids):
            raise ValueError("matrix rows and movie_ids differ in length")
        self.vectorizer = vectorizer
        # Rows are already L2-normalised by the vectorizer; normalising again
        # is a no-op there and makes the dot product a cosine regardless.
        self.matrix = normalize(matrix.tocsr(), norm="l2", copy=True)
        self.movie_ids = np.asarray(movie_ids, dtype=np.int64)
        self.run_date = run_date
        self.top_k = int(top_k)
        self.min_query_terms = int(min_query_terms)

    @classmethod
    def load(cls, stem: str | Path | None = None, config: dict | None = None) -> "SearchIndex":
        import joblib
        import scipy.sparse as sp

        config = config or load_search_config()
        paths = index_files(stem or default_index_path())
        meta = json.loads(paths[".json"].read_text(encoding="utf-8"))
        vectorizer = joblib.load(paths[".joblib"])
        matrix = sp.load_npz(paths[".npz"])
        return cls(vectorizer, matrix, meta["movie_ids"], meta["run_date"],
                   config["top_k"], config["min_query_terms"])

    def query(self, text: str | None) -> SearchResult:
        import numpy as np
        from sklearn.preprocessing import normalize

        if text is None or not str(text).strip():
            return SearchResult(reason=REASON_EMPTY)
        q = self.vectorizer.transform([str(text)])
        n_terms = int(q.nnz)
        if n_terms == 0:
            return SearchResult(reason=REASON_NO_TERMS)
        if n_terms < self.min_query_terms:
            return SearchResult(reason=REASON_TOO_FEW_TERMS, n_terms=n_terms)

        q = normalize(q, norm="l2")
        scores = np.asarray((self.matrix @ q.T).todense()).ravel()
        candidates = np.flatnonzero(scores > 0)
        # Sort by score descending, then movie id ascending; stable and total.
        order = np.lexsort((self.movie_ids[candidates], -scores[candidates]))
        top = candidates[order[: self.top_k]]
        return SearchResult(
            ids=[str(i) for i in self.movie_ids[top]],
            scores=[float(s) for s in scores[top]],
            n_terms=n_terms,
        )


def main() -> None:
    from movierec.storage.factory import get_store

    parser = argparse.ArgumentParser(description="Build the free-text search index from a stored run.")
    parser.add_argument("--run-date", required=True, help="run whose TF-IDF artifacts to index")
    parser.add_argument("--out", default=None, help="index path stem; default env/config search.index_path")
    args = parser.parse_args()

    out = args.out or default_index_path()
    meta = build_search_index(get_store(), args.run_date, out)
    print(f"run_date: {meta['run_date']}")
    print(f"rows x features: {meta['n_rows']} x {meta['n_features']} (nnz {meta['nnz']})")
    for path in index_files(out).values():
        print(f"wrote {path} ({path.stat().st_size:,} B)")


if __name__ == "__main__":
    main()
