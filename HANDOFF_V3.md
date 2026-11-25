# Project handoff brief — movierec v3

*Paste this as the first message of a new session, with `IMPLEMENTATION_V3.md`
attached. `docs/runbook.md` and `docs/comparison.md` are in the repo.*

---

## The project

A content-based movie recommender. A visitor picks a movie and gets 10
similar ones. v3 adds **free-text search**: a visitor types a few words and
gets 10 matching films.

---

## v2 in brief (done 2026-11-01)

- **Live:** https://rueyegqfm4fjepd2vbstidhkti0crznq.lambda-url.us-east-1.on.aws/
  (Lambda image `movierec-api:t4.5-2026-10-26`, 1024 MB, arm64)
- **Weekly batch:** Fargate, Sundays 02:00 ET, ~6 min, ~$0.06 per run. The
  first unattended run passed on 2026-11-01.
- **Alerts:** SNS email on a failed quality gate, a failed task, or more than
  1,000 API requests per hour.
- **Results:** TF-IDF ships as the default. Keyword Jaccard@10 0.0376 vs
  GloVe 0.0241 (random 0.0025).
- **For others:** `make demo` (no AWS, ~64 s first run); README for users;
  `docs/runbook.md` for the operator; `python -m movierec.reload` for
  rollback.
- **Cost:** ~$0.28/month. ECR ~385 MB of the 500 MB free allowance.
- **Data:** frozen in practice; 77,281 films in the catalog.

Full history: v2's `IMPLEMENTATION.md` §9 (in git history) and the runbook.

---

## Decisions already made — do not relitigate unless I ask

1. AWS; no Spark, no cluster; public subnet, no NAT Gateway.
2. Precomputed neighbours, so serving title-based recommendations is a key
   lookup.
3. TMDB keyword tags are held out of all text and indexes. They are
   evaluation labels only.
4. The dense model is **pretrained GloVe**, never "Word2Vec". `word2vec`
   is a legacy identifier, renamed in Phase 7.
5. The README is for users; the runbook is for the operator.
6. v3's small version = T5.1 pinning, T5.2 ECR cleanup, and Phase 6 lexical
   free-text search with a bundled index. Everything else is Phase 7.
7. It's called **free-text search**, never "keyword search".

---

## Where I am right now

*Last updated: 2026-11-24*

**Status:** Phase 5 done (T5.1 `c4ca9aa`, T5.2 `433e17e`). T6.1 chose option (A),
scikit-learn + scipy in the API image (`3d31810`). T6.2's search index and scorer are
committed (`80b106e`), built from run 2026-11-22. T6.4 (API endpoint and page) is done
but uncommitted until `feat: free-text-search-api`; its image `movierec-api:dev` is
built locally only. Nothing new is live: the site still serves `t4.5-2026-10-26`.

**Next:** T6.5 (deploy and document), in week 2 (Nov 23–29). Measure the ECR total
before the push (margin ~12.8 MB, §7 T6.1). T6.3 (evaluation) still comes after T6.5.

**Also in week 2 (not a spec task):** the manual evaluation of v2's recommendations is
finished; analyze it and add the findings to "Evaluation feedback" below (finding 2
onward). Descriptive analysis of the existing judgments only. Scoring with P@10 /
nDCG@10 on these judgments is a new experiment, placed in the Phase 7 backlog
(IMPLEMENTATION_V3.md §6) with the rules to fix before computing.

**Cut line:** applied. T6.2 was committed Nov 24, after ~Nov 20, so T6.3 (evaluation)
moves after T6.5 (Nov 30 – Dec 1).

---

## Task notes

| Task | Status | Commit | Notes |
|---|---|---|---|
| T5.1 Pin dependencies | Done | c4ca9aa | Pinned to the live images' freezes; see Environment |
| T5.2 ECR cleanup | Done | docs: ecr-cleanup | ECR stored before → after: 384.6 → 314.7 MB (of 500) |
| T6.1 Serving decision | Done | docs: search-serving-decision | Decision: **(A) scikit-learn + scipy in the API image** (run D 2026-11-22). Size 172.5 ≤ 185.3 MB and cold load 1.81 ≤ 3 s; ECR margin only 12.8 MB, so measure before T6.5's push (§7) |
| T6.2 Search index + scorer | Done | 80b106e | Index run_date: **2026-11-22**. Option (A): `artifacts/api/search_index.{joblib,npz,json}` (1.98 MB + 25.6 MB + 0.50 MB). 282 tests pass; 5/5 run-D queries match the sklearn reference top 10 (§7) |
| T6.3 Evaluation | | | P@10 vs random: |
| T6.4 API + page | Done | feat: free-text-search-api | `GET /api/query?q=` + "Describe a movie" mode. Local image (not pushed): 172,493,512 B (172.5 MB ≤ 185.3), `sha256:0657d4da68fce4a0724f70b3dd0aca7d6afd73fcb87643b298de51e1805e63f1`, freeze = `constraints/api.txt` (§7). Clean-copy `make demo` check passed ("dream heist" → 10, "zzqx" and empty query → reasons, `/api/similar/27205` OK, page modes / no-match / card click OK) |
| T6.5 Deploy | | | Live tag / digest: |

### T6.2 run-D top 10 (equal to the sklearn reference, 5/5)

- **dream heist:** Notes for a Heist Film · A Heist with Markiplier · Money Heist: The Phenomenon · The Hurricane Heist · The Last Heist · Bank Robbers: The Last Great Heist · Binky Nelson Unpacified · Money Plane · The Bling Ring · Dead Heist
- **survival in space:** Scum · Project Gemini · Final Space · Space Warriors · Closet Space · Tex Montana Will Survive! · Journey to Space · Walking Out · Dead of Winter · Rescue Dawn
- **haunted house:** The Haunted House (1921) · Borley Rectory · Dark House · House of Fears · The House on Pine Street · The Maiden · House of Purgatory · Spooks Run Wild · The Haunted House (1929) · Bhaagamathie
- **time travel romance:** Frequently Asked Questions About Time Travel · Safety Not Guaranteed · Lee's Adventures · SommerHundeSöhne · The History of Time Travel · Back in Time · Timecop · Journey to the Center of Time · Shifter · Kim Possible: A Sitch In Time
- **heist:** A Heist with Markiplier · Notes for a Heist Film · Money Heist: The Phenomenon · The Hurricane Heist · The Last Heist · Bank Robbers: The Last Great Heist · Binky Nelson Unpacified · Money Plane · Dead Heist · White Heat

No Inception for "dream heist": its document (title, tagline, overview, genres) contains neither "dream" nor "heist" (checked on run D), so lexical search can't find it. Note for T6.3, alongside the rare-term pattern from Evaluation feedback #1.

### T6.1 measurements

| | (A) scikit-learn in image | (B) numpy-only index |
|---|---|---|
| Compressed image size | 172.5 MB | 112.8 MB |
| Cold load (local), median / max of 5 | 1.81 s / 1.99 s | 0.43 s / 0.47 s |
| Peak memory (ru_maxrss) | 160.5 MB | 110.8 MB |
| Warm query time, median per query | 4.5–5.2 ms | ~200 ms (unoptimised prototype) |

Run D 2026-11-22, 77,281 films. Size: `docker save \| gzip -c \| wc -c`. Runs: 5 fresh
containers, `--memory=1024m --cpus=0.58`. Queries: 5 fixed queries, 1 warm-up + 50 runs, top 10
over all rows. Rule: (A) if size ≤ 185.3 MB and median cold load ≤ 3 s. Both hold → (A). Full line in §7.

---

## Environment

- venv `movie_rec_venv/`, Python 3.12.5; images are built natively as arm64.
  Docker 28.0.4, AWS CLI 2.37.3, Terraform 1.16.4.
- AWS: IAM user `movierec-dev`, us-east-1, $2 budget alert. The Free plan
  ends 2027-03-10.
- Terraform state is local in `infra/`; `alert_email` is in the gitignored
  `infra/terraform.tfvars`.
- Pinned versions after T5.1 (full locks in `constraints/batch.txt`, `constraints/api.txt`):
  - Base image: `python:3.12.15-slim-trixie@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704`
  - Base deps: pandas 3.0.6, scikit-learn 1.9.1, scipy 1.18.1, joblib 1.6.0,
    pyarrow 25.0.1, pyyaml 6.0.3, kaggle 2.2.4, gensim 4.4.0, psutil 7.2.2,
    boto3 1.43.108
  - `[api]`: flask 3.1.3, rapidfuzz 3.14.6, mangum 0.22.0, a2wsgi 1.10.10
  - API image only: awslambdaric 4.2.0; boto3/botocore 1.43.106 (older than
    batch on purpose, §7 T5.1)

---

## Carried issues and notes

- 4 untagged images in `movierec-batch` from the 2026-10-07 pushes; they add only ~1 MB of
  unique storage, so they were left in place at T5.2 (§7).
- `notes/ecr_usage.py` (gitignored) measures ECR stored bytes. The spec checks the ECR
  total before every push, so consider moving it to `scripts/` before T6.5.

## Evaluation feedback

Findings from manual evaluation of the live recommendations. Fixes are Phase 7
work: each one changes Keyword Jaccard@10, so it is an experiment to name and
approve before running (§0 rule 7).

### 1. Single rare terms pull in unrelated obscure films (TF-IDF, 2026-11-17)

Query: Inception, "Similar details" (TF-IDF). What each neighbour shares with it,
read from the raw overviews:

| # | Neighbour | Shared terms | Votes |
|---|---|---|---|
| 1 | Inception: The Cobol Job | "inception", "cobb" (genuine tie-in) | 290 |
| 2 | Now More than Ever: The History of Chicago | "inception" as in "from their inception in 1967" | 11 |
| 3 | The Wilding | "offered a chance" | 14 |
| 4 | The Fountain of Youth | "offered a chance" | 12 |
| 5 | Prince | "thief", "commits", same genres (loosely fair) | 14 |
| 6 | One Careful Owner | "offered a chance" | 27 |

Causes:
- A rare word gets a high IDF, so sharing it counts as strong evidence even when
  it is used in a different sense (the query's title word in an unrelated overview).
- `ngram_range: [1, 2]` with stop words removed turns stock phrasing into a rare
  bigram ("offered chance"), which alone pulled in three of six neighbours.
- Obscure films have one-sentence overviews; after L2 normalisation one shared
  rare term dominates a short document.

Removing the title from the document would not fix it on its own: Inception's
overview also contains "inception".

Candidate Phase 7 experiments (cheapest first):
1. Popularity floor or prior (vote_count), alongside the planned popularity baseline.
2. Unigrams only, or a higher `min_df` for bigrams.
3. BM25 or length-normalised weighting in place of plain TF-IDF cosine.

Free-text search reuses this TF-IDF representation and forbids reranking
(Phase 6 rules), so expect the same pattern there; note it in T6.3's write-up.

## Spend

- T5.2: $0 (read and delete calls are free); ECR storage down 69.9 MB.

## For the runbook

- Rebuilding the free-text search index when the data changes (T6.5): with run D's
  artifacts in the local store, `STORAGE_BACKEND=local python -m movierec.index.search
  --run-date D` writes `artifacts/api/search_index.{joblib,npz,json}` (it reads only
  the catalog's `id` column, never `keywords`). Then rebuild the API image
  (`make image-api`), which bundles those files, and deploy a new tag. The index's
  `run_date` should match the run DynamoDB serves (check `search_index.json`).
  `make demo` builds its own index under `demo_artifacts/` and never touches these files.
- How to measure the ECR total: stored bytes (unique layers per repo), not the
  sum of `describe-images` sizes (T5.2, §7).

## Lessons from v3

Week 1 (feedback on the system itself is in `docs/week1_v3_summary.md`):
- Renaming the project folder breaks the venv (activate, script shebangs and the editable
  install all store the old path). Fix: `python3.12 -m venv --clear movie_rec_venv`, then
  `pip install -e ".[dev,api]" -c constraints/batch.txt`.
- The local copies of the live images were the only exact record of what production runs.
  Keep the current live image on the laptop until its replacement is live.
- ECR's total is stored unique layers, not the sum of `describe-images` sizes (1,255 MB vs
  385 MB here).
- Reading the matched documents explains odd results in minutes; do that before guessing at fixes.
- Week 1 slipped to a late catch-up on Nov 24; the cut line, set in advance, made the
  reshuffle a rule to follow rather than a decision to make.

---

## How I want you to work with me

- `IMPLEMENTATION_V3.md` is the source of truth. One `T#.#` task per session;
  don't skip a task's dependencies.
- If reality contradicts the spec, update the guide and log it in §7.
  Don't work around it silently.
- Explain cloud concepts when they come up.
- Push back if I'm over-engineering.
- Flag anything that could cost money before I run it, and tell me before
  adding any experiment.
- Never commit or push. Stop at `git status --short` and the list of files
  to add.