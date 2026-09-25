# `eu-ai-rag`

> **A pre-registered, eval-gated RAG system over the EU AI Act and NIS2 Directive.**
> Built to serve two purposes simultaneously: a Lepanto-internal "what does compliance look like for our customers" tool, and a portfolio piece with a defensible engineering story.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/badge/managed%20with-uv-blueviolet.svg)](https://docs.astral.sh/uv/)

---

## What is this?

A retrieval-augmented generation system over the EU AI Act (Regulation (EU) 2024/1689) and the NIS2 Directive (Directive (EU) 2022/2555), sourced from EUR-Lex. The corpus is structured, machine-readable, and authoritative — and the two regulations reference each other, so a single-corpus RAG is genuinely useful rather than a demo toy.

The interesting thing about this build is not the retrieval. It is the **eval loop**: a golden set, Hit@5 / MRR metrics, an LLM-as-judge faithfulness check, and a CI regression gate that fails a pull request when retrieval quality drops. That loop is the same pattern every serious ML system needs.

For the full build plan, decisions, and architecture, see [`BUILD_PLAN.md`](./BUILD_PLAN.md). For step-by-step build guides, see [`TUTORIAL.md`](./TUTORIAL.md).

## Status

**Phases 0–2 done.** Repo skeleton, EUR-Lex corpus ingestion, and dense retrieval v1 are merged and tested. The eval gate (Phase 4) and FastAPI surface (Phase 3) are next.

## Why EU AI Act + NIS2?

The corpus choice is deliberate. Both regulations apply to Lepanto's target customers — chemical operators, port operators, OT-rich mid-market manufacturers. The AI Act covers high-risk AI systems (Annex III); NIS2 covers cybersecurity of essential entities. Together they define the compliance landscape for industrial AI in critical infrastructure. A RAG over both is the artefact we can hand to a customer and say: *"What's your deployer-obligations question? Here's an answer with citations to specific Articles."*

## Quickstart (dev)

```bash
git clone https://github.com/joelsansana/eu-ai-rag.git
cd eu-ai-rag
uv sync --all-extras
uv run pytest -q
```

You will need Python 3.12 and [`uv`](https://docs.astral.sh/uv/). Docker is required from Phase 3 onwards; the integration tests (LLM-as-judge against MiniMax) need a `MINIMAX_API_KEY` and are skipped by default.

## Repository layout

```
eu-ai-rag/
├── README.md              # This file
├── BUILD_PLAN.md          # What was built and why
├── TUTORIAL.md            # Step-by-step build guide
├── TUTORIAL/              # Per-phase build guides
├── LICENSE                # MIT
├── CONTRIBUTING.md
├── pyproject.toml
├── .github/
│   ├── workflows/ci.yml
│   └── ISSUE_TEMPLATE/ + PULL_REQUEST_TEMPLATE.md
├── src/safety_rag/
│   ├── ingestion/         # EUR-Lex parsing, chunking
│   ├── retrieval/         # embedder, vector store
│   ├── generation/        # LLM client, prompts, citations
│   ├── api/               # FastAPI (Phase 3)
│   └── eval/              # golden set, metrics, judge (Phase 4)
├── scripts/               # download_corpus, build_index
├── tests/
├── evals/                 # golden sets (planned)
└── data/                  # gitignored
```

## Contributing

Issues and pull requests welcome. See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for the conventions. For substantial changes, please open an issue first.

## License

[MIT](./LICENSE) — Joel Sansana, 2026.

## TL;DR — what it is and what it costs

> **One-sentence pitch:** A eval-gated RAG system over the EU AI Act + NIS2 Directive, demonstrating customer-empathy for Lepanto (you build the tool your customers will need) and interview-defensible engineering (you own the full eval loop, with regression gating in CI).
>
> **One-number story:** The entire dev-to-eval cycle (ingestion + retrieval v1 + API + golden-set eval + retrieval v2) runs at **~€5 end-to-end** on MiniMax-M2, with local embeddings (zero API cost for CI). Production hosting at <€30/month for a 1000-query dev load. **Total cold-start dev cost: <€50.**
>
> **Two-corpus strategic note:** EU AI Act + NIS2 is *not* an arbitrary corpus choice. NIS2 covers cybersecurity of essential entities (chemical operators, port operators — your Lepanto target customers); AI Act covers high-risk AI systems (industrial process-control AI likely falls under Annex III). A RAG over these two is **simultaneously (a) a Lepanto-internal "what does compliance look like for our customers" tool, and (b) a portfolio piece that demonstrates engineering judgment**. Win-win.
>
> **Pre-registered failure threshold:** if Phase 5 baseline (vanilla RAG) is **≥0.85 Hit@5**, vanilla RAG is "good enough" for the corpus shape and we publish v1 as the whole story (the lesson becomes "vanilla RAG is fine when you have a structured corpus"). If <0.85, we proceed to v2 and quantify the improvement. This makes the v1→v2 narrative defensible no matter what the numbers turn out to be.

## Decisions (defaults, change if you disagree)

|Decision|Default|Why|
|---|---|---|
|**Corpus**|EU AI Act (Regulation (EU) 2024/1689) + NIS2 Directive (Directive (EU) 2022/2555), sourced from EUR-Lex and the official consolidated texts. Two-regulator scope because the two reference each other — NIS2 is the cybersecurity baseline, AI Act is the AI-system overlay; together they define compliance for industrial AI in critical infrastructure (Lepanto's target customer profile). EUR-Lex is structured, machine-readable, free, and authoritative; total corpus is ~600 pages of regulation plus recitals + annexes. **This is simultaneously a Lepanto-internal "what does compliance look like" tool and a portfolio piece.**|
|**Use case**|**Two-tier:** (1) Lepanto-internal — answer customer questions like "what does AI Act Article 26 (deployer obligations) require from our chemical-port customers?" and "what is a 'significant harm' under AI Act Article 3?"; (2) Job-search demo — answer regulation questions with citations to specific Articles. The Lepanto tier is the *real* use case; the demo tier is the more polished surface area. Both share the same corpus and retrieval layer — only the prompt layer and the eval golden set differ between tiers.|
|**Frameworks**|No LangChain/LlamaIndex. Plain `openai` Python client pointed at the MiniMax endpoint (`base_url=https://api.minimax.io/anthropic/v1`, `MINIMAX_API_KEY`), `sentence-transformers`, `qdrant-client`, `fastapi`|Hand-rolled RAG signals understanding; framework-wrappers signal tutorial-following. MiniMax is OpenAI-compatible (via the Anthropic SDK wrapper), so the same `openai` client code works against both OpenAI and MiniMax — swappable, no code fork. |
|**Embeddings**|`BAAI/bge-small-en-v1.5` (local, CPU-friendly, reproducible)|Eval runs in CI without API cost. English-only is fine — both AI Act and NIS2 are translated authoritative texts in EN. If multilingual recall matters, swap to `bge-m3` (multilingual, slightly slower).|
|**Vector store**|Qdrant via docker-compose (in-memory mode for tests)|Two-service compose file = a real MLOps signal. Qdrant's payload filters make Article-level metadata scoping trivial (filter `article_num=26` for a deployer-obligations-only retrieval).|
|**LLM**|**MiniMax-M2** (default; M3 if available — both available via `MINIMAX_API_KEY`) via the OpenAI-compatible API at `api.minimax.io/anthropic/v1`. Fallback to local Ollama with the same `bge-small-en-v1.5` embeddings for a fully air-gapped demo. Use `MiniMax-M2` as default (proven, ~€5 dev cycle), `MiniMax-M3` if you want the latest benchmark. Context: 200K, output: 130K.|OpenAI-compatible (so the `openai` Python client + env-var pattern stays the same — only `base_url` and `api_key` change). **MiniMax-M2 is €0.30/M input + €1.20/M output — total dev cycle ~€5.** Reasoning_mode on for the LLM-as-judge eval step; off for the user-facing API path (faster, cheaper).|
|**Package mgmt**|`uv`, Python 3.12, `ruff` + `pyright` + `pytest`|Current standard; recruiters notice.|
|**Repo name**|`eu-ai-nis2-rag` (leads with the specific scope, not the generic category)|Owns the niche in search; future-proof if the corpus expands (DORA, Cyber Resilience Act, Data Act all live in the same regulatory neighbourhood).|

## Phase plan (~6 weekends, 40–50 hours)

|Phase|Deliverable|Hours|
|---|---|---|
|0. Setup|Repo skeleton, CI running lint+tests|2–3|
|1. Ingestion|Corpus downloaded, parsed, chunked, versioned|5–6|
|2. Retrieval v1|Dense retrieval working, naive RAG answer|4–5|
|3. API|FastAPI app in Docker, documented|5–6|
|4. **Evaluation**|Golden set, metrics, eval CLI, CI gate|8–10|
|5. Retrieval v2|Hybrid search + reranking, improved numbers|4–5|
|6. Story layer|README, demo, launch post|4–5|

The order matters: evaluation comes _before_ improvements, so your improvements are quantified.

## Phase details

**Phase 1 — Ingestion.** Source from EUR-Lex (CELEX numbers: 32024R1689 for the AI Act, 32022L2555 for NIS2) and the EUR-Lex consolidated-text HTML. EUR-Lex HTML is structured (Articles, Recitals, Annexes, Annex III categories all have stable DOM structure) — parse it cleanly with `selectolax` or `lxml`. **Use the official consolidated text**, not draft versions; draft history is preserved as a future-work item. Extract metadata: `regulation` (ai_act | nis2), `part` (recital | article | annex | chapter), `article_num`, `annex_id`, `celex`, `effective_date`, `lang`. Chunk **structure-aware** by Article boundary (each Article = one chunk if it fits; for long Annex III, chunk by row ~400-600 tokens with 75 overlap). Prepend **contextual chunk headers**: e.g., "Article 26, Regulation (EU) 2024/1689 (AI Act), effective 2 August 2026, Chapter 4 — Deployer obligations" — this is a named technique from Anthropic's context-retrieval work and dramatically improves retrieval precision on structured corpora. Store as JSONL with content hashes for reproducibility. **Unit-test the chunker on edge cases**: Annex III (long lists, table-like), Recitals (short paragraphs, no Article heading), Annex IV technical documentation (very long technical-doc section), cross-references between regulations ("as referred to in Article 5 of Directive (EU) 2022/2555").

**Phase 2 — Retrieval v1 (deliberately naive).** Dense embeddings only (`bge-small-en-v1.5` over the contextual-chunk-header prepended content), top-k=5, straightforward prompt with MiniMax-M2 as the generator. Citations to Article + regulation + recital number. Ship this, warts included. This is your **baseline** and the truth-teller for the pre-registered threshold: if v1 already hits ≥0.85 Hit@5 the structured corpus swallowed the question. **The structured EUR-Lex corpus is the easy mode of RAG** — recitals don't cross-reference, articles have stable structure, vocabulary is consistent — so don't be surprised if vanilla RAG is genuinely good enough.

**Phase 3 — API.** FastAPI with Pydantic schemas: `GET /search` (retrieval only), `POST /ask` (full RAG, streaming via SSE), `GET /health`, `GET /stats`, `GET /golden` (returns the eval golden set as JSON — recruiters love this), `POST /feedback` (record user feedback on a query for the next eval round). Every answer includes cited source chunks (`[{reg, article_num, recital_num, chunk_text, score}]`) — citations aren't decoration, they're your faithfulness mechanism **and the audit trail for the Lepanto tier**. Multi-stage Dockerfile (slim base + non-root user + venv); docker-compose with app + Qdrant. `pytest` against the app with Qdrant in-memory mode. **API auth: a single environment-variable API token**, NOT OAuth — auth is out of scope per the decisions table.

**Phase 4 — Evaluation (the reason this project exists).**

- **Golden set (~40-50 Q→source-chunk pairs + ~10 unanswerable):** generate questions from sampled chunks via MiniMax-M2 (reasoning mode ON, temperature 0.2), manually vet every one — disclose this in the README ("LLM-assisted generation, human-verified"). The **Lepanto tier** golden set has customer-shaped questions ("Does Article 26 require a post-market monitoring plan for our process-control AI?"); the **demo tier** has regulation-shaped questions ("What is the deadline for Article 5 prohibited practices?"). Both tiers' golden sets are stored as JSONL in `evals/golden/`. **The unanswerable set tests honest abstention — every RAG demo needs this and most skip it.**
- **Retrieval metrics:** Hit@5, MRR@10, Recall@10 against the golden set. Hit@5 ≥0.85 tripwire per the TL;DR; fall through to Phase 5 otherwise.
- **Generation metrics (LLM-as-judge with MiniMax-M2):** faithfulness (answer supported by retrieved context? — yes / no / partial, with quoted spans), citation accuracy (does the cited Article actually contain the claim? — yes / no, with the cited-text vs. claim-text delta), abstention rate on unanswerable questions (target ≥0.8 of unanswerable questions get an honest "the corpus doesn't address this" answer rather than a hallucinated attempt).
- **CI gate:** GitHub Action runs the retrieval eval on every PR (local embeddings = zero API cost) and **fails if Hit@5 drops >2 points vs. main**. This is the killer feature — you're regression-testing your retrieval quality like a model. The generation eval runs manually/nightly via repo secret.

**Phase 5 — Retrieval v2.** Add BM25 + dense hybrid with reciprocal rank fusion, then a cross-encoder reranker (`BAAI/bge-reranker-base`) on the top-20. Rerun evals. **README table: v1 vs. v2 metrics.** Expected delta: small (~0.85→0.90, or even flatter than CSB → the structured corpus is already easy). **If v2 doesn't move the needle by ≥0.05 Hit@5, that's the story**: "structured regulatory text + contextual chunk headers + MiniMax-M2 generation = vanilla RAG is good enough; the engineering value was the eval loop, not the retrieval upgrades." Either story is defensible. **The pre-registered threshold in the TL;DR is what makes either story defensible.**

**Phase 6 — Story layer.**

- **README as the centerpiece:** problem, Mermaid architecture diagram (renders natively on GitHub), quickstart (`docker compose up` → query in 30 seconds), eval results table, design decisions with rationale, limitations, future work.
- Demo GIF: 30 seconds of querying the archive. Optional: a thin Streamlit chat frontend (you know Streamlit — 2 hours).
- One LinkedIn post: "How I evaluate a RAG system (not just build one)." Recruiters and hiring managers do read these.

## Repo layout

```
eu-ai-nis2-rag/

├── README.md           (start in Phase 0 — fill as you go, finish after Phase 5)
├── pyproject.toml      (uv-managed; ruff, pyright, pytest)
├── Dockerfile          (multi-stage, slim, non-root user, venv)
├── docker-compose.yml  (app + qdrant)
├── .github/workflows/
│   ├── ci.yml          (lint, types, tests on every PR)
│   └── eval.yml        (retrieval eval + Hit@5 regression gate on every PR)
├── src/safety_rag/
│   ├── ingestion/      # eur_lex.py (CELEX fetcher), chunker (Article-aware, contextual-header), metadata
│   ├── retrieval/      # embedder (bge-small-en-v1.5), vector_store (qdrant client), hybrid_search (BM25 + dense), reranker (bge-reranker-base)
│   ├── generation/     # llm (MiniMax-M2 client, OpenAI-compat), prompts (Lepanto tier + demo tier), citations (Article-level)
│   ├── api/            # main.py, schemas.py, auth (env-var API token)
│   └── eval/           # golden loader (lepanto.jsonl, demo.jsonl), metrics (hit@k, MRR, faithfulness), judge (LLM-as-judge)
├── scripts/
│   ├── download_corpus # EUR-Lex fetcher (CELEX 32024R1689, 32022L2555)
│   ├── build_index     # parses → chunks → embeds → qdrant upsert
│   ├── run_eval        # pytest-like eval runner with regression gate
│   └── run_eval_gen    # nightly generation eval via repo secret
├── evals/
│   ├── golden/
│   │   ├── lepanto.jsonl     (~25 customer-shaped questions)
│   │   ├── demo.jsonl        (~25 regulation-shaped questions)
│   │   └── unanswerable.jsonl (~10 unanswerable, for abstention testing)
│   └── results/              (gitignored; eval outputs / Hit@5 history)
├── tests/                    (unit tests on chunker, retrieval, judge; integration tests with qdrant in-memory)
└── data/                     (gitignored, regeneration documented; raw HTML + intermediate JSONL)

Small touch: log queries and retrieval scores to **SQLite** (you know it already) surfaced via `/stats` — p50/p95 latency, score distributions, top-cited Articles. That's your lightweight monitoring story, and it's honest. **Bonus for the Lepanto tier:** a `GET /cited-articles?since=...` endpoint that returns the Articles most-cited in the last N days — feeds into the Lepanto "what are our customers actually asking about?" dashboard.

## What NOT to build (scope control)

- No fine-tuning, no agents, no Kubernetes, no real auth (env-var API token is fine), no LangChain, no multi-tenant anything.
- **No multi-language UI**. English-only interface; the corpus is the European *English translation*, and that's authoritative for compliance work.
- **No persistent user history.** Queries log to SQLite for monitoring, but no per-user session memory. That stays in downstream Lepanto product.
- **No streaming partial answers — only final SSE events.** Streaming partials adds UX complexity without RAG-engineering value.
- **No fine-tuned embeddings.** `bge-small-en-v1.5` is the baseline; if you want multilingual later, swap to `bge-m3`, but don't fine-tune.
- Every one of these is scope creep that delays shipping. **If it's not on a gap list, it doesn't go in the repo.**

## Known limitations / honest failures

This section is filled in **as the project progresses**, not at the end. The structure below is what *will* go in the final README's "Limitations" section; populate as you discover.

### Corpus limits
- [ ] **EUR-Lex translation drift.** The authoritative version is the EU's 24-language set; the English text is a translation. **Failure mode:** a German-native compliance officer might catch a nuance that the EN chunk misses. *Mitigation:* flag EN provenance in every citation; keep DE/FR/NL/CELEX sidecar as a future-work item.
- [ ] **Draft history not captured.** Consolidated text reflects amendments *through* a date. **Failure mode:** "what did Article 26 say in the original 2024 publication?" is unanswerable. *Mitigation:* tracked as future work; document the as-of date in citations.
- [ ] **Cross-references broken at chunk level.** When Article X of the AI Act says "as referred to in Article 5 of Directive (EU) 2022/2555", the chunk returns X but not always the cross-referenced text. **Failure mode:** retrieval misses the right Article in a multi-hop question. *Mitigation:* a chunk-time expansion step (Phase 5 candidate) that pulls in explicitly-referenced Articles.

### Retrieval limits
- [ ] **Recital-only questions.** Recitals are short, narrative, and don't follow the Article structure. **Failure mode:** dense embeddings treat each Recital as a near-duplicate and rank-order them poorly. *Mitigation:* explicit `part=recital` filter in the Qdrant search payload; surface top-3 Recitals when no Article matches cleanly.
- [ ] **Annex III high-cardinality.** Annex III lists 8 high-risk AI-system categories with examples; chunking by row works but loses the "category A and category B both apply" semantics. *Mitigation:* metadata flag `annex_id=III` + multi-row chunk variant.
- [ ] **Long definitions in early Articles.** Articles 3 and 4 of the AI Act are definition-heavy. **Failure mode:** definitions overlap; embedding space collapses them. *Mitigation:* retrieval-augmented generation for definitions: pull all of Article 3 at query-time, prefix the prompt with `[Use Article 3 of AI Act as the canonical definition source for any term defined there]`.

### Generation limits (LLM-as-judge honesty)
- [ ] **Faithfulness judge is itself a model.** MiniMax-M2 is the judge; bias flows both ways. **Mitigation:** keep ~20% of the golden set as human-judge baseline; if LLM-judge accuracy diverges from human-judge by >10%, surface it in the README.
- [ ] **Citation accuracy requires pulling exact cited text.** The API has to return verbatim text of the cited Article with `display=inline` rendering for the user. *Mitigation:* the `GET /golden` endpoint already exposes this; the user-facing path is consistent.
- [ ] **No conversation memory.** Each `/ask` is a single-turn call. **Mitigation:* document; explicitly out-of-scope per the What NOT section.

### Cost & latency limits
- [ ] **Cold-start latency (first query).** MiniMax-M2 cold-start adds ~2-5 s of latency on the first request of the day. *Mitigation:* keep a warm `httpx.AsyncClient` connection pool; benchmark before shipping.
- [ ] **Peak load characterization.** No load test against 1000 concurrent /ask. *Mitigation:* a Phase 6 stretch goal `locust` load test; document the saturation point.

### Operational limits
- [ ] **No automated EUR-Lex re-fetch.** When a new consolidated text is published (amendment, corrigendum), the corpus doesn't refresh itself. *Mitigation:* a weekly `scripts/download_corpus` cron + content-hash diffing that alerts Joel via dashboard if anything changes.
- [ ] **No GDPR audit log.** Every query is logged, but query-data-retention isn't aligned with GDPR Article 30 records. *Mitigation:* document in README; out-of-scope for Lepanto-internal demo, in-scope if this ever ships to real customers.

*This section gets revisited at every phase boundary and at the end of Phase 6.*