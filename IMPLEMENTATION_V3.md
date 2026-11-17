# Movie Recommender v3 — Implementation Guide (AWS)

**Status:** spec for v3, small version; target end of November 2026. Builds on
v2 (live since 2026-10-14; first unattended weekly run 2026-11-01).<br>
**Budget:** ~1 hour/day on weekdays, plus Sundays. No work Saturdays.<br>
**Audience:** the author, and Claude Code<br>
**Scope:** Phase 5 (one prerequisite) and Phase 6 (free-text search). Phase 7
is the later backlog and is not scheduled.

---

## 0. How to use this document

**Humans:** read §1–§3 once, then work through §4–§5 one task per session.
v2's guide stays as history; new deviations go in §7 here.

**Claude Code:** this file is the source of truth. Rules:

1. Do exactly one task per session. A task is one `T#.#` block.
2. Never start a task whose **Depends on** is not yet committed.
3. Every task ends with its **Done when** check passing. If it fails, fix it in the same session; do not proceed.
4. All infrastructure is Terraform in `infra/`. No new AWS services.
5. Never commit credentials, `.env`, `terraform.tfvars`, or anything in `data/` except `data/sample/`.
6. If reality disagrees with this spec, **update this file in the same commit** and note it under §7.
7. **Experiments are listed in their task.** Any other run, metric or comparison needs the author's OK first; every experiment is described in the write-up.
8. Never commit or push. Stop at `git status --short` and the list of files to add.

---

## 1. Goal

### User story

> A visitor types a few words describing a film ("dream heist", "survival in
> space"), and sees 10 matching movies with poster, year, rating, and a
> one-line overview.

### Engineering goal

- Add **lexical free-text search** to the live site, scored with v2's existing TF-IDF representation.
- Keep v2's properties: under $1/month, a slim serving image, no matrix work beyond what the measurement in T6.1 justifies.
- Evaluate search on held-out labels (TMDB keyword tags), not on vibes.
- Make builds reproducible before the next image push: pinned dependencies, base image by digest.

### Explicit non-goals

- **No semantic search.** No GloVe or sentence-embedding query vectors in v3's small version.
- **No combined "movie + description" queries.**
- **No TMDB keyword tags in any index or document.** They stay the held-out label (CLAUDE.md hard constraint).
- **No new AWS services** (no OpenSearch, no API Gateway, no NAT Gateway).
- **No index move into the pipeline yet.** The search index is bundled in the API image like the title index; safe while the dataset is frozen. The move is Phase 7.

**Reverses v2's §1 non-goal** "No free-text semantic search", for lexical search only.

---

## 2. Target architecture

v2's batch side is unchanged. v3 adds a search index, built once from a stored run and shipped inside the API image, plus one new endpoint.

```
                    ┌──────────────────────────────┐
   EventBridge      │  ECS Fargate task (batch)    │
   Scheduler   ───► │  2 vCPU / 4 GiB, ~6 min      │   unchanged
   (Sun 02:00 ET)   │  public subnet, no NAT       │
                    │  ingest → gates → TF-IDF +   │
                    │  GloVe → top-K → evaluate →  │
                    │  S3 → DynamoDB → current.json│
                    └──────────────┬───────────────┘
            ┌──────────────────────┼──────────────────────┐
            ▼                      ▼                      ▼
     ┌─────────────┐       ┌──────────────┐       ┌──────────────┐
     │     S3      │       │   DynamoDB   │       │  CloudWatch  │
     │ run         │       │ movie items  │       │ + SNS email  │
     │ artifacts   │       │ + top-10     │       │   alerts     │
     └──────┬──────┘       └──────┬───────┘       └──────────────┘
            │ vectorizer +        │
            │ matrix of run D     │ GetItem / BatchGetItem
            ▼                     │
  ┌─────────────────────────┐     │
  │ NEW search-index build  │     │
  │ laptop, once per data   │     │
  │ change:                 │     │
  │ movierec.index.search   │     │
  │   --run-date D          │     │
  └───────────┬─────────────┘     │
              │ search_index.*    │
              ▼ bundled at build  ▼
  ┌──────────────────────────────────────┐
  │  Lambda (container, arm64, 1024 MB)  │
  │  + Function URL                      │
  │  /api/search   title suggestions     │
  │  /api/similar  precomputed top-10    │
  │  /api/query    NEW free-text search  │
  │  /             page, NEW mode        │
  │                "Describe a movie"    │
  └──────────────────┬───────────────────┘
                     ▼
                  browser
```

### How a free-text query is served (`/api/query?q=`)

1. **Vectorise:** the query text becomes a TF-IDF vector, using run D's vocabulary and IDF weights. That's the same analyzer the documents went through, or a reimplementation proven equal to it.
2. **Score:** cosine similarity against the bundled matrix, in memory, over all 77K films, keeping the top 10. There's no database scan.
3. **Hydrate:** one `BatchGetItem` fetches the display fields, exactly as `/api/similar` does.

The index loads once per cold start. `/api/health` reports the index's `run_date`, so any drift from DynamoDB is visible.

### Why each new piece

| Decision | Why not something else |
|---|---|
| Index built from an **S3 run** | Local `artifacts/` holds only sample runs (262 films) |
| Built **on the laptop**, not in the batch task | Keeps the batch task unchanged for the small version. Safe while the dataset is frozen; moving it into the pipeline is Phase 7 |
| **Bundled in the image**, not loaded from S3 at cold start | No S3 permission or extra latency for the API; the same reasoning as the title index |
| Scored **in Lambda memory** | A sparse dot product over 77K films takes milliseconds. A managed search service (OpenSearch) would cost tens of dollars a month, roughly 100× the whole bill |
| Display fields **from DynamoDB** | One source of truth for what cards show, shared with `/api/similar` |
| Serving library **chosen in T6.1** | scikit-learn in the image costs size and cold start; numpy only costs tokeniser parity. Measure, then decide |

### Cost target

| Line | Monthly |
|---|---|
| Batch (DynamoDB writes, Fargate, S3), as measured in v2 | ~$0.28 |
| ECR, kept inside the 500 MB free allowance; total checked before every push | $0 (~$0.10/GB-month above it) |
| Lambda, Function URL, EventBridge, CloudWatch, SNS | $0 (free tier) |
| DynamoDB reads, on-demand, including NEW: 1 `BatchGetItem` per search | a fraction of a cent at current traffic |
| **Total** | **~$0.28, unchanged** |

The only real cost risk is image size. Option (A) adds scikit-learn and scipy to the API image, and that's why T6.1 measures it before deciding.

## 3. Conventions (in addition to v2's CLAUDE.md)

- **Naming:** "free-text search" in code, docs and the page. Never "keyword search": `keywords` is the held-out TMDB field.
- **Pins:** exact `==` versions in `pyproject.toml` (base and `[api]`), and both Dockerfiles `FROM python:3.12-slim@sha256:…`.
- **ECR budget:** check the ECR total before every push; never push without an apply plan; one new tag per rebuild (v2 rule).
- **API guard:** `tests/test_api_imports.py` keeps banning pandas, pyarrow and gensim. Any newly allowed library (numpy, or scikit-learn and scipy) is changed in the test deliberately and logged in §7.
- **New config keys** in `config/default.yaml`: `search.top_k`, `search.min_query_terms`, and the index path.

---

## 4. Phase 5 — Prerequisites

### T5.1 — Pin dependencies and the base image
**Commit:** `build: pin-dependencies`

Pin every dependency to the version **currently in the live images**: run `pip freeze` inside both locally built images, with no AWS. Pin both base images by digest. Check whether a newer Mangum removes the `get_event_loop` DeprecationWarning; adopt it only if the tests pass.

**Done when:** `pytest -q` passes; each image built twice gives an identical `pip freeze`; nothing is pushed.

### T5.2 — ECR cleanup
**Depends on:** T5.1
**Commit:** `docs: ecr-cleanup` (§7 only)

Delete the three old `movierec-api:t4.4-*` images; the live `t4.5-2026-10-26` stays. Record the ECR total before and after.

**Done when:** `aws ecr describe-images` lists only the live API image plus the batch images; the live site still answers; the totals are in §7.

---

## 5. Phase 6 — Free-text search (small version)

⚠️ **Rules — fixed before running, not changed after seeing results:**

- **Index input:** v2's fitted TF-IDF vectorizer and matrix from one named S3 run. No refitting, no extra fields.
- **Query processing** is the vectorizer's own analyzer, or a reimplementation proven equal to it (T6.2).
- **Ranking** is cosine similarity, top `search.top_k` = 10. No reranking tweaks.

### T6.1 — Serving decision (measure, then decide)
**Depends on:** T5.2
**Commit:** `docs: search-serving-decision`

Measure both options locally:

- **(A)** scikit-learn and scipy in the API image, loading `vectorizer.joblib` and `matrix.npz` at cold start.
- **(B)** a compact numpy-only index: vocabulary, IDF and the sparse matrix, with the analyzer reimplemented.

For each, record the compressed image size, the local cold-load time, peak memory and the warm query time over 77K films.

**Decision rule (set now):** choose (A) if its image fits the ECR allowance alongside one rollback image *and* its cold load is ≤ 3 s; otherwise choose (B).

**Done when:** a §7 line has both sets of measurements and the decision.

### T6.2 — Search index and scorer
**Depends on:** T6.1
**Commit:** `feat: free-text-search-index`

`python -m movierec.index.search --run-date D` reads the run's vectorizer, matrix, row index and catalog through `get_store()`. It writes `artifacts/api/search_index.*`, which records its `run_date`. The scorer returns the top 10 ids and scores for a query string.

- An empty or fully out-of-vocabulary query returns no results and a reason. It never returns random films.
- With option (B): a parity test shows the reimplemented analyzer gives the same vector as `vectorizer.transform` on 50 seeded sample queries.

**Done when:** `pytest -q` passes, including the parity test, the out-of-vocabulary case, determinism, and a guard that the builder never reads a `keywords` column. For 5 sample queries, the top 10 matches a scikit-learn reference ranking.

### T6.3 — Evaluation (new experiment)
**Depends on:** T6.2
**Commit:** `docs: free-text-search-evaluation`

**Queries:** a seeded sample of TMDB keyword tags, each with at least 20 tagged films. A film counts as relevant if it carries the tag; tags are not in the indexed text, so this is held out.

**Report:** P@10 against a random floor, plus the share of queries with zero results.

**Note:** tag words can appear in overviews. That's a legitimate lexical match, so state it.

**Done when:** `docs/search_eval.md` has the numbers and the sample design.

**Cut line:** if T6.2 isn't committed by about Nov 20, T6.3 moves after T6.5.

### T6.4 — API endpoint and page
**Depends on:** T6.2
**Commit:** `feat: free-text-search-api`

- **Endpoint:** `GET /api/query?q=` returns 10 results with display fields, hydrated with `BatchGetItem` as `/api/similar` does.
- **Page:** a mode switch, "Find by title" / "Describe a movie", with empty and no-match states. Use `textContent` only, and never show keywords.
- **Demo:** `make demo` keeps working, with search served from a local index built from the demo artifacts.

**Done when:** `pytest -q` passes (endpoint tests, plus the guard test updated per §3); `make demo` shows search results for "dream heist" on a clean copy.

### T6.5 — Deploy and document
**Depends on:** T6.4
**Commit:** `feat: free-text-search-live`

1. Check the ECR total.
2. Build the pinned image, push one new tag, set `api_image_tag`, apply (expect only `image_uri` to change).
3. Warm the site, then record the cold-start Init Duration.
4. README: a short "Describe a movie" line under Run it yourself.
5. Runbook: rebuilding the search index when the data changes.

**Done when:** 3 live queries return 10 results each in under 2 s warm; `/api/search` and `/api/similar` are unchanged; the README and runbook are updated; §7 has the tag, digest and ECR total.

---

## 6. Phase 7 — Later (not scheduled)

- Move the title and search indexes into the pipeline (`catalog/{run_date}/…`, loaded via `current.json`), plus a search tie-break by `vote_count` ("Dark Night" vs "The Dark Knight"; "incep" → Harry Potter).
- Skip the DynamoDB load when unchanged; full overview in DynamoDB (cost test first).
- Rename the `word2vec` identifiers (needs a pipeline run plus a full reload, using `movierec.reload`).
- Least-privilege IAM; import Lambda's ECR repository policy.
- Entrypoint timeout 45m → 15m; `run_date` semantics; a "no successful run in 8 days" alarm.
- Measure a real reload's time; a `schedule_enabled` variable.
- Popularity baseline; the "matched on" explanation; collaborative filtering / hybrid; GloVe or sentence-embedding queries.

---


## 7. Deviations from this spec

Append whenever reality differs. Date, task ID, what changed, why.

- *(none yet)*