# Build Plan — `eu-ai-rag`

> **A pre-registered, eval-gated RAG system over the EU AI Act and NIS2 Directive.**
> This document is the **build plan** for the project: what was built, in what order, why each phase exists, and what was learned. If you want to *build it yourself* from scratch, follow the per-phase guides in [`TUTORIAL/`](./TUTORIAL/). If you want to *understand the artefact* as it stands today, read this.

| | |
|---|---|
| **Status** | In active development — Phases 0–2 complete; Phases 3–6 on the roadmap |
| **License** | [MIT](./LICENSE) |
| **Repository** | <https://github.com/joelsansana/eu-ai-rag> |
| **Author** | Joel Sansana |
| **Maintainer** | Number One 🖖 (assisted) |

---

## Why this project exists

Two reasons, deliberately intertwined:

1. **A Lepanto-internal "what does compliance look like for our customers" tool.** Lepanto's target customers — chemical operators, port operators, OT-rich mid-market manufacturers — sit at the intersection of the EU AI Act (Annex III high-risk systems) and NIS2 (essential-entity cybersecurity baseline). A RAG over both is the artifact we can hand to a customer during a sales conversation and ask, *"What's your deployer-obligations question? Here's an answer with citations to specific Articles."*
2. **A portfolio piece with a defensible engineering story.** The interesting thing about this build is not the retrieval — it's the **eval loop**: a golden set, Hit@5 / MRR metrics, an LLM-as-judge faithfulness check, and a CI regression gate that fails a pull request when retrieval quality drops. That loop is the same pattern every serious ML system needs.

The corpus was chosen to serve both reasons simultaneously. EU AI Act + NIS2 are *structured, machine-readable, authoritative, free*, and they reference each other — that last property is what makes a single-corpus RAG genuinely useful rather than a demo toy.

---

## The build, in one table

| Phase | Deliverable | Hours | Status |
|---|---|---|---|
| 0 | Repo skeleton, CI running lint + tests | 2–3 | ✅ Done |
| 1 | EUR-Lex corpus downloaded, parsed, chunked, versioned | 5–6 | ✅ Done |
| 2 | Dense retrieval v1, naive RAG answering | 4–5 | ✅ Done |
| 3 | FastAPI app, Docker, multi-stage build | 5–6 | ⬜ Next |
| 4 | Golden set, metrics, eval CLI, CI regression gate | 8–10 | ⬜ Planned |
| 5 | Hybrid search + reranker (conditional on Phase 4) | 4–5 | ⬜ Conditional |
| 6 | README polish, demo, launch post | 4–5 | ⬜ Stretch |

**The order is deliberate.** Phase 4 (evaluation) lands *before* Phase 5 (retrieval upgrades) so any improvement is quantified against a measured baseline. Phase 5 itself is conditional: if Phase 4 says the baseline already meets the pre-registered threshold (≥0.85 Hit@5), Phase 5 is no longer an upgrade, it's a story — *"vanilla RAG was already enough, the engineering value was the eval loop."*

---

## Architecture (as built)

```text
┌─────────────────────┐    ┌────────────────────────┐    ┌─────────────────────┐
│  EUR-Lex corpus     │    │  Ingestion pipeline    │    │  Qdrant vector store │
│  (CELEX 32024R1689, │───▶│  - HTML parser         │───▶│  - Article-level    │
│   CELEX 32022L2555) │    │  - Article-aware       │    │    chunking         │
│                     │    │    chunker             │    │  - payload metadata │
│                     │    │  - contextual headers  │    │  (reg, article,     │
│                     │    │  - content-hash vcsion │    │   recital, celex)   │
└─────────────────────┘    └────────────────────────┘    └─────────────────────┘
                                                                  │
                                                                  ▼
┌─────────────────────┐    ┌────────────────────────┐    ┌─────────────────────┐
│  FastAPI app        │◀───│  Retrieval layer       │◀───│  Embedder           │
│  GET /search        │    │  - dense (bge-small)   │    │  BAAI/bge-small-    │
│  POST /ask          │    │  - hybrid (planned:    │    │  en-v1.5 (local,    │
│  GET /health        │    │    BM25 + dense + RRF) │    │  zero API cost)     │
│  GET /stats         │    │  - reranker (planned:  │    │                     │
│  GET /golden        │    │    bge-reranker-base)  │    │                     │
└─────────────────────┘    └────────────────────────┘    └─────────────────────┘
            │
            ▼
┌─────────────────────┐
│  Generation layer   │
│  - MiniMax-M2       │
│    (OpenAI-compat)  │
│  - Article-level    │
│    citations        │
│  - two-tier prompts │
│    (Lepanto / demo) │
└─────────────────────┘
```

---

## Decisions (defaults, change with reason)

| Decision | Choice | Why |
|---|---|---|
| **Corpus** | EU AI Act + NIS2, sourced from EUR-Lex CELEX 32024R1689 and 32022L2555 | Structured, machine-readable, free, authoritative; the two reference each other |
| **Embeddings** | `BAAI/bge-small-en-v1.5` (local) | CPU-friendly, reproducible, zero API cost in CI |
| **Vector store** | Qdrant via docker-compose | Payload filters make Article-level metadata scoping trivial |
| **LLM** | MiniMax-M2 via OpenAI-compatible API | Same `openai` Python client works against both OpenAI and MiniMax; ~€5 per dev cycle |
| **Frameworks** | None. Hand-rolled | Framework wrappers signal tutorial-following; hand-rolled signals understanding |
| **Package mgmt** | `uv`, Python 3.12, `ruff` + `pyright` + `pytest` | Current standard |
| **Pre-registered threshold** | Phase 4 must hit ≥0.85 Hit@5 before Phase 5 is justified | Makes the v1→v2 narrative defensible regardless of which way the numbers go |

---

## What this build demonstrates

| Capability | Where it lives |
|---|---|
| **Eval-gated ML.** Golden set, retrieval metrics, LLM-as-judge, CI regression gating | Phase 4 (planned) |
| **Structured-corpus retrieval.** Article-aware chunking, contextual chunk headers, payload filters | `src/safety_rag/ingestion/`, `src/safety_rag/retrieval/` |
| **Citations as a faithfulness mechanism.** Every answer links to a specific Article + CELEX number | `src/safety_rag/generation/prompts.py` |
| **Honest abstention.** Unanswerable questions return "the corpus doesn't address this" rather than a hallucinated attempt | Phase 4 (planned) |
| **Production-shape packaging.** `pyproject.toml`, multi-stage Docker, slim base, non-root user | Phase 3 (planned) |
| **Cost discipline.** ~€5 dev cycle end-to-end; local embeddings in CI | Whole stack |

---

## Repository layout

```text
eu-ai-rag/
├── README.md                    # Public overview (the one GitHub renders)
├── BUILD_PLAN.md                # This file — what was built and why
├── TUTORIAL.md                  # Step-by-step build guide (learner-oriented)
├── TUTORIAL/                    # Per-phase build guides
│   ├── phase-0-setup.md
│   ├── phase-1-ingestion.md
│   ├── phase-2-retrieval-v1.md
│   ├── phase-3-api.md
│   ├── phase-4-eval.md
│   └── phase-5-retrieval-v2.md
├── LICENSE                      # MIT
├── CONTRIBUTING.md              # How to file issues, PR conventions
├── pyproject.toml               # uv-managed; ruff, pyright, pytest
├── Dockerfile                   # multi-stage, slim, non-root user, venv (Phase 3)
├── docker-compose.yml           # app + qdrant (Phase 3)
├── .github/
│   ├── workflows/
│   │   ├── ci.yml               # lint, types, tests on every PR
│   │   └── eval.yml             # (planned) retrieval eval + Hit@5 regression gate
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   └── PULL_REQUEST_TEMPLATE.md
├── src/safety_rag/
│   ├── ingestion/               # eur_lex.py, chunker, metadata
│   ├── retrieval/               # embedder, vector_store (hybrid + reranker planned)
│   ├── generation/              # llm client, prompts, citations
│   ├── api/                     # FastAPI app (Phase 3)
│   └── eval/                    # golden loader, metrics, judge (Phase 4)
├── scripts/
│   ├── download_corpus.py       # EUR-Lex CELEX fetcher
│   └── build_index.py           # parse → chunk → embed → qdrant upsert
├── evals/
│   ├── golden/                  # (planned) lepanto.jsonl, demo.jsonl, unanswerable.jsonl
│   └── results/                 # gitignored
├── tests/                       # unit + integration (Qdrant in-memory)
└── data/                        # gitignored; raw + intermediate, regeneration documented
```

---

## What is NOT being built (scope control)

- No LangChain / LlamaIndex. The `openai` Python client is enough.
- No fine-tuning, no agents, no Kubernetes, no multi-tenant anything.
- No real auth (env-var API token is sufficient for a portfolio piece).
- No persistent user history; queries log to SQLite for monitoring only.
- No multilingual UI; corpus is the official English translation.
- No fine-tuned embeddings; `bge-small-en-v1.5` baseline, swap to `bge-m3` only if multilingual recall is required.

---

## Known limitations (filled as the project progresses)

This section is populated *as* limits are discovered, not at the end. The full honesty log lives in [`BUILD_PLAN_LIMITATIONS.md`](./BUILD_PLAN_LIMITATIONS.md) (planned). Pre-discovered:

- **Translation drift.** The English EUR-Lex text is a translation; a native-DE compliance officer may catch nuances the EN chunk misses.
- **Draft history not captured.** "What did Article 26 say in the original 2024 publication?" is unanswerable from the consolidated text alone.
- **Cross-references broken at chunk level.** When Article X of the AI Act references Article 5 of NIS2, retrieval of X does not always pull in the cross-referenced text.
- **Recital-only questions.** Recitals are short, narrative, and dense embeddings treat them as near-duplicates.
- **LLM-as-judge is itself a model.** Bias flows both ways; ~20% of the golden set is human-judge baseline.

---

## How to read this project

- **You want to use the running system:** wait for Phase 3 (API) + Phase 4 (eval gate) to land, then `docker compose up` and query. Not ready today.
- **You want to extend the system:** read [`TUTORIAL.md`](./TUTORIAL.md), then the relevant `TUTORIAL/phase-N.md` for the area you're changing.
- **You want to understand the engineering decisions:** this file. Every choice links back to a "why."
- **You want to evaluate the engineering:** look at the test suite under `tests/`, the chunker in `src/safety_rag/ingestion/chunker.py`, and the eval design (Phase 4, planned).
- **You want to file an issue:** see [CONTRIBUTING.md](./CONTRIBUTING.md).

---

## Status legend

- ✅ Done — code merged, tests green
- ⬜ Planned — designed, not started
- 🔄 In progress
- ⛔ Blocked

The build is currently at **Phases 0–2 done**. Phase 3 (FastAPI) is next; Phase 4 (eval) is the centrepiece and the reason this project exists.

---

*This build plan is the public-facing artefact for the `eu-ai-rag` project. The internal Obsidian-vault plan (with private context) remains in `~/Documents/Notas/`; this file is what a GitHub visitor sees. Maintained by Joel Sansana.*
