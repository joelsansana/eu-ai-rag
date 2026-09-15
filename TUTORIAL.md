# TUTORIAL

> Step-by-step build guide for `eu-ai-rag`. You are reading this because you want to **build the project yourself**, using these notes as a tutorial rather than running a generated scaffold.

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
| 0 | Repo skeleton + CI running lint+tests | 2–3 | ⬜ |
| 1 | Corpus downloaded, parsed, chunked, versioned | 5–6 | ⬜ |
| 2 | Dense retrieval + naive RAG answering | 4–5 | ⬜ |
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
- **GitHub CLI authenticated** — already done (`gh auth status` should show `joelsansana`)
- **A Qdrant container** — comes via `docker-compose.yml` in Phase 3, but for Phase 2 testing you can run it standalone with `docker run -p 6333:6333 qdrant/qdrant`

## The end-state repo layout

```
eu-ai-rag/
├── README.md                    # The centerpiece (existing — don't touch for now)
├── TUTORIAL.md                  # This file
├── TUTORIAL/
│   ├── phase-0-setup.md
│   ├── phase-1-ingestion.md
│   ├── phase-2-retrieval-v1.md
│   ├── phase-3-api.md
│   ├── phase-4-eval.md
│   └── phase-5-retrieval-v2.md
├── pyproject.toml               # uv-managed; ruff, pyright, pytest
├── Dockerfile                   # multi-stage, slim, non-root user, venv
├── docker-compose.yml           # app + qdrant
├── .github/workflows/
│   ├── ci.yml                   # lint, types, tests on every PR
│   └── eval.yml                 # retrieval eval + Hit@5 regression gate on every PR
├── src/safety_rag/
│   ├── ingestion/               # eur_lex.py, chunker, metadata
│   ├── retrieval/               # embedder, vector_store, hybrid_search, reranker
│   ├── generation/              # llm (MiniMax-M2), prompts, citations
│   ├── api/                     # main.py, schemas.py, auth
│   └── eval/                    # golden loader, metrics, judge
├── scripts/
│   ├── download_corpus          # EUR-Lex fetcher (CELEX 32024R1689, 32022L2555)
│   ├── build_index              # parse → chunk → embed → qdrant upsert
│   ├── run_eval                 # pytest-like eval runner
│   └── run_eval_gen             # nightly generation eval
├── evals/
│   ├── golden/
│   │   ├── lepanto.jsonl        # ~25 customer-shaped questions
│   │   ├── demo.jsonl           # ~25 regulation-shaped questions
│   │   └── unanswerable.jsonl   # ~10 unanswerable (for abstention test)
│   └── results/                 # gitignored; eval outputs / Hit@5 history
├── tests/                       # unit + integration (Qdrant in-memory)
└── data/                        # gitignored; raw HTML + intermediate JSONL
```

The layout matches the README's "Repo layout" section, so you can cross-reference as you build.

## Decision overrides

The decisions in the README are starting points. If you disagree with any, change it **in the README first**, then build the phase that depends on it. Common override candidates:

| Decision | Likely override scenario |
|---|---|
| Corpus | Add DORA / Cyber Resilience Act / Data Act later (schema already accommodates this) |
| Embeddings | Swap `bge-small-en-v1.5` → `bge-m3` if multilingual recall matters for your Lepanto customers |
| LLM | Swap MiniMax-M2 → GPT-4o or Claude if you want a different cost/quality profile |
| Vector store | Swap Qdrant → pgvector if you'd rather stay in Postgres-land |
| Package mgmt | Drop `uv` for `poetry`/`pdm` if you have a preference |

Whatever you change, document it in the README's decisions table with the new rationale. The eval loop will surface whether the change helped or hurt.

## What NOT to build (scope reminders)

- No LangChain / LlamaIndex (per the decisions table — hand-rolled RAG signals understanding)
- No fine-tuning, no agents, no Kubernetes, no real auth (env-var API token only)
- No multi-language UI, no persistent user history, no streaming partials
- No fine-tuned embeddings

If something is tempting you into scope creep, write it down in the README's "Known limitations / honest failures" section as a future-work item and move on.

---

**Ready?** Start with [`TUTORIAL/phase-0-setup.md`](TUTORIAL/phase-0-setup.md).
