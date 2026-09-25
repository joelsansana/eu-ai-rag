# TUTORIAL

> Step-by-step build guide for `eu-ai-rag`. You are reading this because you want to **build the project yourself**, using these notes as a tutorial rather than running a generated scaffold.
>
> **Last corrected 2026-09-25** by Number One against the actual code: phases 0–2 verified against `src/safety_rag/`, `tests/`, `scripts/`, `pyproject.toml`, and the GitHub Actions workflow. The phase map below reflects the build's current state (0–2 ✅ done, 3–5 ⬜ planned).

This file is the entry point. The per-phase files (`TUTORIAL/phase-0-setup.md`, `phase-1-ingestion.md`, …) are the actual build instructions. Read this first, then go in order.

---

## What you are building

A eval-gated RAG system over the **EU AI Act** (Regulation (EU) 2024/1689) and **NIS2 Directive** (Directive (EU) 2022/2555), sourced from EUR-Lex. Two-tier use case:

1. **Lepanto tier** — internal "what does compliance look like for our port/chemical customers" tool
2. **Demo tier** — job-search portfolio piece (regulator-shaped questions, citations to specific Articles)

The defining feature is the **eval loop**: a golden set + Hit@5/MRR metrics + LLM-as-judge + a CI regression gate. The retrieval upgrades (Phase 5) are conditional on the eval saying they're needed.

## How to use these guides

- **One phase per weekend.** The README's phase table says ~40–50 hours total over ~6 weekends.
- **Commit at the end of each phase.** Each guide ends with an explicit commit step.
- **Run the verify block before committing.** If the verify block fails, the phase isn't done — fix it before moving on.
- **Read the pitfalls section.** They are common mistakes I expect you to make. Reading them first makes them less likely.
- **You can override decisions.** The decisions table is a starting point, not law. If you disagree, change it in the README *before* starting the phase that depends on it.

## The phase map

| Phase | Deliverable | Hours | Status |
|---|---|---|---|
| 0 | Repo skeleton + CI running lint+tests | 2–3 | ✅ |
| 1 | Corpus downloaded, parsed, chunked, versioned | 5–6 | ✅ |
| 2 | Dense retrieval + naive RAG answering | 4–5 | ✅ |
| 3 | FastAPI app in Docker | 5–6 | ⬜ |
| 4 | Golden set + metrics + CI regression gate | 8–10 | ⬜ |
| 5 | Hybrid search + reranker (conditional) | 4–5 | ⬜ |

**Order matters.** Phase 4 (eval) comes *before* Phase 5 (retrieval upgrades) so any improvement is quantified.

## Shared setup (one-time)

You will need:

- **Python 3.12** — `python --version` should report 3.12.x
- **uv** — the package manager ([install instructions](https://docs.astral.sh/uv/)); `uv --version`
- **Docker + Docker Compose** — for Phase 3 onwards; `docker --version && docker compose version`
- **A MiniMax API key** — for the LLM-as-judge and generation calls; the dev cycle is ~€5 total
- **GitHub CLI authenticated** — `gh auth status` should show your handle
- **A Qdrant container** — for Phase 2 testing you can run it standalone with `docker run -p 6333:6333 qdrant/qdrant`. Phase 3 will introduce `docker-compose.yml` for app + Qdrant.

## The end-state repo layout

```text
eu-ai-rag/
├── README.md                    # Public overview
├── BUILD_PLAN.md                # What was built and why
├── TUTORIAL.md                  # This file
├── TUTORIAL/
│   ├── phase-0-setup.md         ✅
│   ├── phase-1-ingestion.md     ✅
│   ├── phase-2-retrieval-v1.md  ✅
│   ├── phase-3-api.md           ⬜
│   ├── phase-4-eval.md          ⬜
│   └── phase-5-retrieval-v2.md  ⬜
├── pyproject.toml               # uv-managed; ruff, pyright, pytest
├── LICENSE                      # MIT
├── CONTRIBUTING.md
├── .github/
│   ├── workflows/
│   │   ├── ci.yml               # lint, types, tests on every PR
│   │   └── eval.yml             # (planned) retrieval eval + Hit@5 regression gate
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE.md
├── src/safety_rag/
│   ├── __init__.py              # package entry point: main()
│   ├── ingestion/               # eur_lex.py (parser), chunker.py (Article-aware + long-annex split)
│   ├── retrieval/               # embedder.py (bge-small-en-v1.5), vector_store.py (Qdrant)
│   ├── generation/              # llm.py (MiniMax-M2 via OpenAI SDK), prompts.py (RAG_PROMPT + helpers)
│   ├── api/                     # ask.py (end-to-end ask() function — Phase 3 wraps this in FastAPI)
│   └── eval/                    # (planned) golden loader, metrics, judge
├── scripts/
│   ├── download_corpus.py       # EUR-Lex/Cellar fetcher (CELEX 32024R1689, 32022L2555)
│   └── build_index.py           # load data/processed/*.jsonl → upsert to Qdrant
├── evals/
│   ├── golden/                  # (planned) lepanto.jsonl, demo.jsonl, unanswerable.jsonl
│   └── results/                 # gitignored; eval outputs / Hit@5 history
├── tests/                       # unit + integration (Qdrant + real LLM); integration skipped by default
└── data/                        # gitignored; raw HTML + intermediate JSONL
```

**Layout notes (corrections from the original tutorial):**

- `scripts/run_eval` and `scripts/run_eval_gen` are planned for Phase 4, not yet present.
- `Dockerfile` and `docker-compose.yml` are planned for Phase 3, not yet present.
- `.github/workflows/eval.yml` is planned for Phase 4, not yet present.
- `src/safety_rag/api/main.py`, `schemas.py`, and `auth.py` are planned for Phase 3. Phase 2 only has `api/ask.py`.
- `src/safety_rag/eval/` is planned for Phase 4, not yet present.

## Decision overrides

The decisions in the README are starting points. If you disagree with any, change it **in the README first**, then build the phase that depends on it. Common override candidates:

| Decision | Likely override scenario |
|---|---|
| Corpus | Add DORA / Cyber Resilience Act / Data Act later (the parser's `CELEX_TO_REGULATION` map accommodates this) |
| Embeddings | Swap `bge-small-en-v1.5` → `bge-m3` if multilingual recall matters for your Lepanto customers |
| LLM | Swap MiniMax-M2 → GPT-4o or Claude if you want a different cost/quality profile |
| Vector store | Swap Qdrant → pgvector if you'd rather stay in Postgres-land |
| Package mgmt | Drop `uv` for `poetry`/`pdm` if you have a preference |
| Token estimation | Swap the regex-based `_estimate_tokens` for `tiktoken` if you need byte-level accuracy |

Whatever you change, document it in the README's decisions table with the new rationale. The eval loop will surface whether the change helped or hurt.

## What NOT to build (scope reminders)

- No LangChain / LlamaIndex (per the decisions table — hand-rolled RAG signals understanding)
- No fine-tuning, no agents, no Kubernetes, no real auth (env-var API token only)
- No multi-language UI, no persistent user history, no streaming partials
- No fine-tuned embeddings
- **No `reasoning` parameter on `generate()`.** The user-facing path is faster without it; if you need reasoning (e.g. for the Phase 4 LLM-as-judge), add a separate function instead of a flag.

If something is tempting you into scope creep, write it down in the README's "Known limitations / honest failures" section as a future-work item and move on.

---

**Ready?** Start with [`TUTORIAL/phase-0-setup.md`](TUTORIAL/phase-0-setup.md).
