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

*Last updated: <date>*

**Status:** <current task, what's committed, what's live>

**Next:** <next task and session date>

**Cut line:** if T6.2 isn't committed by ~Nov 20, T6.3 (evaluation) moves
after T6.5.

---

## Task notes

| Task | Status | Commit | Notes |
|---|---|---|---|
| T5.1 Pin dependencies | | | |
| T5.2 ECR cleanup | | | ECR before → after: |
| T6.1 Serving decision | | | Decision: |
| T6.2 Search index + scorer | | | Index run_date: |
| T6.3 Evaluation | | | P@10 vs random: |
| T6.4 API + page | | | |
| T6.5 Deploy | | | Live tag / digest: |

### T6.1 measurements

| | (A) scikit-learn in image | (B) numpy-only index |
|---|---|---|
| Compressed image size | | |
| Cold load (local) | | |
| Peak memory | | |
| Warm query time | | |

---

## Environment

- venv `movie_rec_venv/`, Python 3.12.5; images are built natively as arm64.
  Docker 28.0.4, AWS CLI 2.37.3, Terraform 1.16.4.
- AWS: IAM user `movierec-dev`, us-east-1, $2 budget alert. The Free plan
  ends 2027-03-10.
- Terraform state is local in `infra/`; `alert_email` is in the gitignored
  `infra/terraform.tfvars`.
- Pinned versions after T5.1: <fill in>

---

## Carried issues and notes

- <open items found during v3>

## Spend

- <per task, when AWS was used>

## For the runbook

- <anything to add to `docs/runbook.md` at T6.5, e.g. rebuilding the search
  index>

## Lessons from v3

- <fill in at the end of each week>

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