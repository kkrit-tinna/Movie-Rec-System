import re
import shutil
import subprocess
from pathlib import Path

import boto3
import pandas as pd
import pytest
from flask import Flask
from moto import mock_aws
from rapidfuzz import fuzz, utils
from sklearn.feature_extraction.text import TfidfVectorizer

from movierec.api import __main__ as api_main
from movierec.api.app import METHODS, create_app, load_config
from movierec.index.search import (
    REASON_EMPTY,
    REASON_NO_TERMS,
    REASON_TOO_FEW_TERMS,
    SearchIndex,
)
from movierec.index.titles import build_title_index, default_titles_path, load_title_index
from movierec.storage.dynamo import DynamoReader, DynamoWriter, build_items

TABLE = "movies"
K = 10
TITLES = [
    "Inception", "The Godfather", "Toy Story", "Interstellar", "The Dark Knight",
    "Insomnia", "Memento", "Tenet", "Up!", "Up", "Alien", "Aliens", "Heat",
    "Heat", "M",
]
RECOMMENDATION_FIELDS = {"id", "title", "year", "score", "rating", "poster_path", "overview", "genres"}
SEARCH_RUN_DATE = "2026-11-22"


def _catalog() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "id": [100 + i for i in range(len(TITLES))],
            "title": TITLES,
            "release_date": [f"{1990 + i}-01-01" for i in range(len(TITLES))],
            "poster_path": [f"/p{i}.jpg" for i in range(len(TITLES))],
            "vote_average": [7.0 + i / 10 for i in range(len(TITLES))],
            "overview": [f"overview of {t}, " + "x" * 300 for t in TITLES],
            "genres": "Action, Science Fiction",
        }
    )
    return df.astype({"title": "string", "release_date": "string", "poster_path": "string"})


def _neighbours(catalog: pd.DataFrame, reverse: bool) -> pd.DataFrame:
    """Ten neighbours each; word2vec ranks them in the opposite order so the
    two methods are distinguishable."""
    ids = list(catalog["id"])
    records = []
    for movie_id in ids:
        others = [m for m in ids if m != movie_id][:K]
        if reverse:
            others = others[::-1]
        for rank, nid in enumerate(others, start=1):
            records.append((movie_id, rank, nid, round(0.9 - rank * 0.05, 4)))
    return pd.DataFrame(records, columns=["movie_id", "rank", "neighbour_id", "score"])


def _search_index(catalog: pd.DataFrame, min_query_terms: int = 1) -> SearchIndex:
    """A small real index: every film's text has "space station" plus its
    title, so "space" matches all 15 films and "inception" only one."""
    docs = [f"space station {t}" for t in catalog["title"]]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(docs)
    config = load_config()["search"]
    return SearchIndex(
        vectorizer, matrix, list(catalog["id"]), SEARCH_RUN_DATE,
        config["top_k"], min_query_terms,
    )


@pytest.fixture
def client(monkeypatch):
    for name, value in {
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_SESSION_TOKEN": "testing",
        "AWS_DEFAULT_REGION": "us-east-1",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("AWS_PROFILE", raising=False)

    catalog = _catalog()
    with mock_aws():
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        ddb.create_table(
            TableName=TABLE,
            KeySchema=[{"AttributeName": "movie_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "movie_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        items = build_items(
            catalog,
            {"tfidf": _neighbours(catalog, False), "word2vec": _neighbours(catalog, True)},
        )
        DynamoWriter(TABLE, client=ddb).write(items)
        app = create_app(
            DynamoReader(TABLE, client=ddb),
            build_title_index(catalog),
            "tfidf",
            search_index=_search_index(catalog),
        )
        yield app.test_client()


class TestPage:
    def test_index_is_html_with_search_box(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert response.mimetype == "text/html"
        assert '<input id="q"' in response.get_data(as_text=True)

    def test_page_never_uses_innerhtml_or_absolute_api_urls(self, client):
        page = client.get("/").get_data(as_text=True)
        script = page[page.index("<script>"):]
        assert "innerHTML" not in script
        assert "onerror" not in page
        assert '"/api/search' in script and '"/api/similar/' in script
        assert '"/api/query?q=' in script
        assert "lambda-url" not in page and "localhost" not in page

    def test_page_never_shows_keywords(self, client):
        # Keywords are the held-out evaluation label, never a shown reason.
        page = client.get("/").get_data(as_text=True)
        assert "keyword" not in page[page.index("<script>"):].lower()

    def test_mode_switch_and_free_text_naming(self, client):
        page = client.get("/").get_data(as_text=True)
        assert re.search(r'data-mode="title"[^>]*>Find by title<', page)
        assert re.search(r'data-mode="describe"[^>]*>Describe a movie<', page)
        assert '<input id="describe-q"' in page
        assert "free-text search" in page.lower()
        assert "keyword search" not in page.lower()

    def test_every_query_reason_has_plain_words(self, client):
        script = client.get("/").get_data(as_text=True)
        script = script[script.index("<script>"):]
        for reason in (REASON_EMPTY, REASON_NO_TERMS, REASON_TOO_FEW_TERMS):
            assert re.search(rf"\b{reason}: \"[A-Z][^\"]+\"", script), reason

    def test_tmdb_link_built_from_digit_only_id(self, client):
        page = client.get("/").get_data(as_text=True)
        fn = re.search(r"function tmdbUrl\(id\) \{.*?\n  \}", page, re.S).group(0)
        assert "/^[0-9]+$/.test(id)" in fn
        assert '"https://www.themoviedb.org/movie/" + id' in fn
        # The only href set on the page comes from tmdbUrl, on the hero.
        assert page.count(".href =") == 1 and "more.href = url;" in page
        assert 'more.rel = "noopener noreferrer";' in page
        if shutil.which("node") is None:
            pytest.skip("node not installed; static checks only")
        script = fn + """
            for (const id of ["27205", 155, "27205x", "1/../x", "", "javascript:1"])
                console.log(JSON.stringify(tmdbUrl(id)));
        """
        out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
        assert out.stdout.split() == [
            '"https://www.themoviedb.org/movie/27205"',
            '"https://www.themoviedb.org/movie/155"',
            "null", "null", "null", "null",
        ]

    def test_example_films_exist_in_title_index(self, client):
        path = Path(default_titles_path())
        if not path.exists():
            pytest.skip(f"title index not built at {path}")
        page = client.get("/").get_data(as_text=True)
        examples = dict(re.findall(r'class="chip" data-id="(\d+)">([^<]+)<', page))
        assert len(examples) == 4
        by_id = {t["id"]: t["title"] for t in load_title_index(path)}
        assert {i: by_id.get(i) for i in examples} == examples


class TestHealth:
    def test_reports_title_count_and_search_run_date(self, client):
        assert client.get("/api/health").get_json() == {
            "status": "ok",
            "titles": len(TITLES),
            "search_index_run_date": SEARCH_RUN_DATE,
        }

    def test_run_date_null_without_search_index(self):
        client = create_app(None, [], "tfidf", config=load_config()).test_client()
        assert client.get("/api/health").get_json()["search_index_run_date"] is None


class TestQuery:
    def test_ten_results_with_display_fields_in_index_order(self, client):
        body = client.get("/api/query?q=space").get_json()
        assert body["reason"] is None and body["q"] == "space"
        results = body["results"]
        assert len(results) == K
        assert all(set(r) == RECOMMENDATION_FIELDS for r in results)
        expected = _search_index(_catalog()).query("space")
        assert [r["id"] for r in results] == expected.ids
        assert [r["score"] for r in results] == expected.scores
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True) and scores[-1] > 0

    def test_display_fields_come_from_the_item(self, client):
        (only,) = client.get("/api/query?q=inception").get_json()["results"]
        assert {k: v for k, v in only.items() if k != "score"} == {
            "id": "100",
            "title": "Inception",
            "year": 1990,
            "rating": 7.0,
            "poster_path": "/p0.jpg",
            "overview": ("overview of Inception, " + "x" * 300)[:200],
            "genres": ["Action", "Science Fiction"],
        }

    def test_fewer_than_ten_matches_returns_fewer(self, client):
        body = client.get("/api/query?q=inception").get_json()
        assert body["reason"] is None
        assert [r["id"] for r in body["results"]] == ["100"]

    @pytest.mark.parametrize("url", ["/api/query", "/api/query?q=", "/api/query?q=%20%20"])
    def test_empty_query(self, client, url):
        response = client.get(url)
        assert response.status_code == 200
        assert response.get_json()["results"] == []
        assert response.get_json()["reason"] == REASON_EMPTY

    @pytest.mark.parametrize("q", ["zzzqqq", "the and of"])
    def test_no_known_terms(self, client, q):
        response = client.get(f"/api/query?q={q}")
        assert response.status_code == 200
        assert response.get_json()["results"] == []
        assert response.get_json()["reason"] == REASON_NO_TERMS

    def test_too_few_known_terms(self):
        index = _search_index(_catalog(), min_query_terms=2)
        client = create_app(None, [], "tfidf", config=load_config(), search_index=index).test_client()
        response = client.get("/api/query?q=space")
        assert response.status_code == 200
        assert response.get_json()["results"] == []
        assert response.get_json()["reason"] == REASON_TOO_FEW_TERMS

    def test_hydrates_with_one_batch_get_and_none_when_no_results(self):
        calls = []

        class CountingReader:
            def get_item(self, movie_id):
                raise AssertionError("/api/query must not call get_item")

            def batch_get(self, ids):
                calls.append(list(ids))
                return [{"movie_id": i, "title": "T"} for i in ids]

        index = _search_index(_catalog())
        client = create_app(CountingReader(), [], "tfidf", config=load_config(), search_index=index).test_client()
        assert len(client.get("/api/query?q=space").get_json()["results"]) == K
        assert calls == [index.query("space").ids]
        client.get("/api/query?q=zzzqqq")
        client.get("/api/query?q=")
        assert len(calls) == 1

    def test_index_path_from_config_then_env(self, monkeypatch):
        from movierec.api.app import search_index_path
        from movierec.index.search import INDEX_PATH_ENV

        config = load_config()
        monkeypatch.delenv(INDEX_PATH_ENV, raising=False)
        assert search_index_path(config) == config["search"]["index_path"]
        monkeypatch.setenv(INDEX_PATH_ENV, "/tmp/elsewhere/search_index")
        assert search_index_path(config) == "/tmp/elsewhere/search_index"

    def test_503_without_search_index(self):
        client = create_app(None, [], "tfidf", config=load_config()).test_client()
        assert client.get("/api/query?q=space").status_code == 503


class TestSearch:
    def test_best_match_first(self, client):
        results = client.get("/api/search?q=incep").get_json()["results"]
        assert results[0] == {"id": "100", "title": "Inception", "year": 1990}

    def test_case_and_word_order_ignored(self, client):
        results = client.get("/api/search?q=GODFATHER the").get_json()["results"]
        assert results[0]["title"] == "The Godfather"

    def test_one_char_query_returns_empty(self, client):
        assert client.get("/api/search?q=i").get_json() == {"results": []}
        assert client.get("/api/search?q= ").get_json() == {"results": []}
        assert client.get("/api/search").get_json() == {"results": []}

    def test_limit_respected_and_clamped(self, client):
        assert len(client.get("/api/search?q=alien&limit=3").get_json()["results"]) == 3
        assert len(client.get("/api/search?q=alien&limit=0").get_json()["results"]) == 1
        # Every title but "M": one char is under min_fuzzy_length_ratio of "alien".
        results = client.get("/api/search?q=alien&limit=500").get_json()["results"]
        assert len(results) == len(TITLES) - 1
        assert "M" not in [r["title"] for r in results]

    def test_result_shape(self, client):
        for r in client.get("/api/search?q=alien").get_json()["results"]:
            assert set(r) == {"id", "title", "year"}


class TestExactMatch:
    def test_one_char_exact_title_found(self, client):
        results = client.get("/api/search?q=m").get_json()["results"]
        assert [r["title"] for r in results] == ["M"]

    def test_one_char_non_title_is_empty(self, client):
        assert client.get("/api/search?q=z").get_json() == {"results": []}

    def test_exact_match_first_even_when_fuzzy_scores_tie(self, client):
        # "Up!" and "Up" both process to "up" and score 100; "Up!" comes first
        # in the catalog, so fuzzy order alone would put it first.
        assert fuzz.WRatio("up", utils.default_process("Up!")) == fuzz.WRatio("up", "up") == 100
        results = client.get("/api/search?q=  UP ").get_json()["results"]
        assert [r["title"] for r in results[:2]] == ["Up", "Up!"]

    def test_remakes_newest_first(self, client):
        results = client.get("/api/search?q=heat").get_json()["results"]
        years = sorted((1990 + i for i, t in enumerate(TITLES) if t == "Heat"), reverse=True)
        assert len(years) == 2
        assert [(r["title"], r["year"]) for r in results[:2]] == [("Heat", y) for y in years]

    def test_no_duplicates(self, client):
        for q in ("heat", "up", "inception", "alien"):
            ids = [r["id"] for r in client.get(f"/api/search?q={q}&limit=20").get_json()["results"]]
            assert len(ids) == len(set(ids))

    def test_exact_matches_respect_limit(self, client):
        assert len(client.get("/api/search?q=heat&limit=1").get_json()["results"]) == 1


class TestSimilar:
    @pytest.mark.parametrize("method,reverse", [("tfidf", False), ("word2vec", True)])
    def test_ten_in_neighbour_order_with_v1_fields(self, client, method, reverse):
        body = client.get(f"/api/similar/100?method={method}").get_json()
        assert body["found"] is True
        assert body["method"] == method
        assert body["query"]["id"] == "100"
        recs = body["recommendations"]
        expected = [str(101 + i) for i in range(K)]
        assert [r["id"] for r in recs] == (expected[::-1] if reverse else expected)
        assert all(set(r) == RECOMMENDATION_FIELDS for r in recs)
        assert [r["score"] for r in recs] == [round(0.9 - i * 0.05, 4) for i in range(1, K + 1)]

    def test_display_fields(self, client):
        first = client.get("/api/similar/100").get_json()["recommendations"][0]
        assert first == {
            "id": "101",
            "title": "The Godfather",
            "year": 1991,
            "score": 0.85,
            "rating": 7.1,
            "poster_path": "/p1.jpg",
            "overview": ("overview of The Godfather, " + "x" * 300)[:200],
            "genres": ["Action", "Science Fiction"],
        }

    def test_query_block_has_hero_display_fields(self, client):
        # Same fields as a card, minus score; from the GetItem already made.
        assert client.get("/api/similar/100").get_json()["query"] == {
            "id": "100",
            "title": "Inception",
            "year": 1990,
            "rating": 7.0,
            "poster_path": "/p0.jpg",
            "overview": ("overview of Inception, " + "x" * 300)[:200],
            "genres": ["Action", "Science Fiction"],
        }

    def test_hero_fields_need_no_extra_dynamodb_calls(self):
        calls = []

        class CountingReader:
            def get_item(self, movie_id):
                calls.append("get_item")
                return {"movie_id": movie_id, "title": "A", "similar_tfidf": [{"id": "2", "score": 0.5}]}

            def batch_get(self, ids):
                calls.append("batch_get")
                return [{"movie_id": i, "title": "B"} for i in ids]

        app = create_app(CountingReader(), [], "tfidf", config=load_config())
        assert app.test_client().get("/api/similar/1").status_code == 200
        assert calls == ["get_item", "batch_get"]

    def test_default_method_is_the_configured_one(self, client):
        assert client.get("/api/similar/100").get_json()["method"] == "tfidf"

    def test_n_clamped(self, client):
        def count(n):
            return len(client.get(f"/api/similar/100?n={n}").get_json()["recommendations"])

        assert count(3) == 3
        assert count(0) == 1
        assert count(-5) == 1
        assert count(500) == K  # clamped to 20, but an item holds only k neighbours
        assert count("abc") == K  # not an int -> default

    def test_unknown_method_is_400_listing_allowed(self, client):
        response = client.get("/api/similar/100?method=bert")
        assert response.status_code == 400
        error = response.get_json()["error"]
        assert all(m in error for m in METHODS)

    def test_unknown_id_is_404(self, client):
        response = client.get("/api/similar/999999")
        assert response.status_code == 404
        assert response.get_json() == {"found": False, "message": "Movie not found"}


class TestFuzzyLengthRule:
    """Fuzzy candidates far shorter than the query are dropped; exact matches
    are not. Search never touches the reader, so none is needed here."""

    TITLES = [
        "Inception", "I", "P", "IO", "The Crack: Inception", "Her", "The Godfather",
        "Godfather", "Matrix", "The Matrix", "Ma", "99", "M", "Up", "It", "Titanic",
    ]

    def _search(self, query, ratio=None):
        config = load_config()
        if ratio is not None:
            config["api"]["min_fuzzy_length_ratio"] = ratio
        titles = [{"id": str(i), "title": t, "year": 2000} for i, t in enumerate(self.TITLES)]
        client = create_app(None, titles, "tfidf", config=config).test_client()
        return [r["title"] for r in client.get(f"/api/search?q={query}&limit=20").get_json()["results"]]

    def test_short_titles_dropped_from_fuzzy_tail(self):
        assert self._search("incep")[0] == "Inception"
        assert not {"I", "P", "M"} & set(self._search("incep"))
        assert "IO" not in self._search("inception")
        assert "Her" not in self._search("godfather")

    def test_threshold_is_inclusive(self):
        # "the matrix 1999" is 15 chars: 6 >= 0.4 * 15 is kept, 2 is not.
        results = self._search("the matrix 1999")
        assert {"The Matrix", "Matrix"} <= set(results)
        assert not {"Ma", "99"} & set(results)

    def test_short_exact_matches_still_first(self):
        assert self._search("m") == ["M"]
        assert self._search("up")[0] == "Up"
        assert self._search("it")[0] == "It"
        assert self._search("inception")[0] == "Inception"

    def test_ratio_comes_from_config(self):
        assert "I" in self._search("incep", ratio=0.0)
        # 5.0 drops every fuzzy candidate here (longest title 20 < 5 * 5),
        # but not the exact match.
        assert self._search("incep", ratio=5.0) == []
        assert self._search("inception", ratio=5.0) == ["Inception"]


class TestCreateApp:
    def test_rejects_unknown_default_method(self):
        with pytest.raises(ValueError, match="bert"):
            create_app(reader=None, titles=[], default_method="bert")


class TestMain:
    @pytest.fixture
    def run_calls(self, monkeypatch):
        calls = []
        monkeypatch.setattr(Flask, "run", lambda self, **kw: calls.append(kw))
        monkeypatch.setattr(api_main, "DynamoReader", lambda: object())
        monkeypatch.setattr(api_main, "load_title_index", lambda path: [{"id": "1", "title": "X", "year": None}])
        monkeypatch.setattr(api_main, "load_search_index", lambda config: None)
        monkeypatch.delenv(api_main.PORT_ENV, raising=False)
        monkeypatch.delenv(api_main.HOST_ENV, raising=False)
        return calls

    def test_local_dev_server_on_loopback_port_from_config_debug_off(self, run_calls):
        api_main.main(["--local"])
        assert run_calls == [{"host": "127.0.0.1", "port": 8000, "debug": False}]

    def test_port_flag_and_env(self, run_calls, monkeypatch):
        monkeypatch.setenv(api_main.PORT_ENV, "8123")
        api_main.main(["--local"])
        api_main.main(["--local", "--port", "9000"])
        assert [c["port"] for c in run_calls] == [8123, 9000]

    def test_host_env_overrides_loopback(self, run_calls, monkeypatch):
        monkeypatch.setenv(api_main.HOST_ENV, "0.0.0.0")
        api_main.main(["--local"])
        assert run_calls[0]["host"] == "0.0.0.0"

    def test_requires_local(self, run_calls):
        with pytest.raises(SystemExit):
            api_main.main([])
        assert run_calls == []
