"""Free-text search index and scorer (T6.2). Uses the committed 5k sample."""
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from movierec.data.ingest import load, prepare
from movierec.embedders.tfidf import TfidfEmbedder
from movierec.index import search
from movierec.index.search import SearchIndex, build_search_index, index_files
from movierec.storage.local import LocalStore

SAMPLE_CSV = Path(__file__).resolve().parents[1] / "data" / "sample" / "movies_5k.csv"
RUN_DATE = "2026-01-01"
CONFIG = {"top_k": 10, "min_query_terms": 1}
QUERIES = ["dream heist", "survival in space", "haunted house", "time travel romance", "heist",
           "a family drama", "war"]


@pytest.fixture(scope="module")
def sample_run(tmp_path_factory):
    """A sample run in a LocalStore, laid out as the pipeline writes it
    (the catalog keeps its keywords column, as in a real run)."""
    root = tmp_path_factory.mktemp("artifacts")
    store = LocalStore(root=str(root))
    catalog = prepare(load(str(SAMPLE_CSV)))
    assert "keywords" in catalog.columns
    embedder = TfidfEmbedder()
    embedder.fit(catalog["document"].tolist(), catalog["id"].tolist())
    embedder.save(store, RUN_DATE)
    buf = io.BytesIO()
    catalog.to_parquet(buf, index=False)
    store.put_bytes(f"catalog/{RUN_DATE}/catalog.parquet", buf.getvalue())
    return store, embedder, catalog


@pytest.fixture(scope="module")
def built(sample_run, tmp_path_factory):
    store, _, _ = sample_run
    stem = tmp_path_factory.mktemp("api") / "search_index"
    meta = build_search_index(store, RUN_DATE, stem)
    return stem, meta


@pytest.fixture(scope="module")
def index(built):
    stem, _ = built
    return SearchIndex.load(stem, config=CONFIG)


def _reference(embedder, text, k=10):
    """scikit-learn reference: vectorizer.transform + cosine, id ascending on ties."""
    scores = cosine_similarity(embedder.vectorizer.transform([text]), embedder.matrix).ravel()
    ids = np.asarray(embedder.movie_ids, dtype=np.int64)
    cand = np.flatnonzero(scores > 0)
    order = np.lexsort((ids[cand], -scores[cand]))[:k]
    return [str(i) for i in ids[cand][order]]


class TestBuilder:
    def test_writes_three_files_and_records_run_date(self, built):
        stem, meta = built
        assert all(p.exists() for p in index_files(stem).values())
        assert meta["run_date"] == RUN_DATE
        assert SearchIndex.load(stem, config=CONFIG).run_date == RUN_DATE

    def test_movie_ids_follow_matrix_rows(self, built, sample_run):
        _, meta = built
        _, embedder, _ = sample_run
        assert meta["movie_ids"] == [int(i) for i in embedder.movie_ids]
        assert meta["n_rows"] == embedder.matrix.shape[0]
        assert meta["n_features"] == embedder.matrix.shape[1]

    def test_copies_run_artifacts_byte_for_byte(self, built, sample_run):
        stem, _ = built
        store, _, _ = sample_run
        files = index_files(stem)
        assert files[".joblib"].read_bytes() == store.get_bytes(f"tfidf/{RUN_DATE}/vectorizer.joblib")
        assert files[".npz"].read_bytes() == store.get_bytes(f"tfidf/{RUN_DATE}/matrix.npz")

    def test_build_is_deterministic(self, sample_run, tmp_path):
        store, _, _ = sample_run
        a, b = tmp_path / "a" / "idx", tmp_path / "b" / "idx"
        build_search_index(store, RUN_DATE, a)
        build_search_index(store, RUN_DATE, b)
        for suffix in search.SUFFIXES:
            assert index_files(a)[suffix].read_bytes() == index_files(b)[suffix].read_bytes()

    def test_rejects_catalog_that_does_not_match_row_index(self, sample_run, tmp_path):
        store, _, catalog = sample_run
        other = LocalStore(root=str(tmp_path / "store"))
        for name in ("vectorizer.joblib", "matrix.npz", "row_index.parquet"):
            other.put_bytes(f"tfidf/{RUN_DATE}/{name}", store.get_bytes(f"tfidf/{RUN_DATE}/{name}"))
        buf = io.BytesIO()
        catalog.iloc[1:].to_parquet(buf, index=False)
        other.put_bytes(f"catalog/{RUN_DATE}/catalog.parquet", buf.getvalue())
        with pytest.raises(ValueError, match="catalog"):
            build_search_index(other, RUN_DATE, tmp_path / "idx")


class TestKeywordsGuard:
    def test_catalog_columns_exclude_keywords(self):
        assert "keywords" not in search.CATALOG_COLUMNS

    def test_builder_never_reads_a_keywords_column(self, sample_run, tmp_path, monkeypatch):
        store, _, _ = sample_run
        calls = []
        real = pd.read_parquet

        def spy(*args, **kwargs):
            calls.append(kwargs.get("columns"))
            return real(*args, **kwargs)

        monkeypatch.setattr(pd, "read_parquet", spy)
        build_search_index(store, RUN_DATE, tmp_path / "idx")
        assert len(calls) == 2  # row index and catalog
        for columns in calls:
            assert columns is not None, "a parquet read without columns= reads keywords"
            assert "keywords" not in columns


class TestEmptyAndOutOfVocabulary:
    @pytest.mark.parametrize("text", ["", "   ", "\n\t", None])
    def test_empty_query_returns_nothing_with_reason(self, index, text):
        result = index.query(text)
        assert result.ids == [] and result.scores == []
        assert result.reason == search.REASON_EMPTY

    @pytest.mark.parametrize("text", ["zzqxv qwrtyp", "xylophonequux", "the and of"])
    def test_out_of_vocabulary_query_returns_nothing_with_reason(self, index, text):
        result = index.query(text)
        assert result.ids == [] and result.scores == []
        assert result.reason == search.REASON_NO_TERMS

    def test_below_min_query_terms_returns_nothing_with_reason(self, built):
        stem, _ = built
        strict = SearchIndex.load(stem, config={"top_k": 10, "min_query_terms": 50})
        result = strict.query("a family drama")
        assert 0 < result.n_terms < 50
        assert result.ids == [] and result.reason == search.REASON_TOO_FEW_TERMS


class TestScorer:
    def test_returns_top_k_positive_scores_best_first(self, index):
        result = index.query("a family drama")
        assert result.reason is None
        assert len(result.ids) == CONFIG["top_k"]
        assert all(s > 0 for s in result.scores)
        assert result.scores == sorted(result.scores, reverse=True)
        assert all(isinstance(i, str) for i in result.ids)

    def test_top_k_from_config(self, built):
        stem, _ = built
        small = SearchIndex.load(stem, config={"top_k": 3, "min_query_terms": 1})
        assert len(small.query("a family drama").ids) == 3

    def test_default_config_has_search_keys(self):
        config = search.load_search_config()
        assert config["top_k"] == 10
        assert config["min_query_terms"] >= 1
        assert config["index_path"]

    @pytest.mark.parametrize("text", QUERIES)
    def test_deterministic_across_calls_and_loads(self, built, index, text):
        stem, _ = built
        again = SearchIndex.load(stem, config=CONFIG)
        first = index.query(text)
        assert index.query(text) == first
        assert again.query(text) == first

    @pytest.mark.parametrize("text", QUERIES)
    def test_matches_sklearn_reference_ranking(self, index, sample_run, text):
        _, embedder, _ = sample_run
        assert index.query(text).ids == _reference(embedder, text)

    def test_ties_broken_by_id_ascending(self):
        texts = ["space station survival", "space station survival",
                 "space station survival", "a quiet village romance"]
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(texts)
        idx = SearchIndex(vectorizer, matrix, [30, 7, 12, 1], "t", top_k=10, min_query_terms=1)
        result = idx.query("space survival")
        assert result.ids == ["7", "12", "30"]  # equal scores, id ascending; row 4 scores 0

    def test_zero_score_films_are_never_padded_in(self):
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(["alpha beta", "gamma delta", "epsilon zeta"])
        idx = SearchIndex(vectorizer, matrix, [1, 2, 3], "t", top_k=10, min_query_terms=1)
        assert idx.query("alpha").ids == ["1"]


def test_scorer_import_does_not_load_pandas():
    """The API (T6.4) imports SearchIndex; the slim image has no pandas."""
    code = "import sys, movierec.index.search; sys.exit('pandas' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


class TestCli:
    """`python -m movierec.index.search --config ...` as `make demo` runs it."""

    def test_config_overlay_sets_store_and_index_path(self, sample_run, tmp_path, monkeypatch):
        store, _, _ = sample_run
        out = tmp_path / "demo_api" / "search_index"
        overlay = tmp_path / "overlay.yaml"
        overlay.write_text(
            f"storage:\n  local_root: {store.root}\nsearch:\n  index_path: {out}\n"
        )
        monkeypatch.delenv(search.INDEX_PATH_ENV, raising=False)
        search.main(["--storage", "local", "--config", str(overlay), "--run-date", RUN_DATE])
        assert all(p.exists() for p in index_files(out).values())
        assert SearchIndex.load(out, config=CONFIG).run_date == RUN_DATE

    def test_sample_config_keeps_demo_index_out_of_artifacts(self):
        repo = Path(__file__).resolve().parents[1]
        default = search.load_config()["search"]["index_path"]
        demo = search.load_config(str(repo / "config" / "sample.yaml"))
        assert default == "artifacts/api/search_index"
        assert demo["search"]["index_path"].startswith(demo["storage"]["local_root"] + "/")
        assert demo["search"]["index_path"] != default
