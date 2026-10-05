# Phase 4 — Evaluation (the reason this project exists)

> Golden set + retrieval metrics + generation metrics (LLM-as-judge) + CI regression gate. ~8–10 hours.

## Goal

- `evals/golden/lepanto.jsonl` — ~25 customer-shaped Q→source-chunk pairs + metadata, **human-verified**
- `evals/golden/demo.jsonl` — ~25 regulation-shaped Q→source-chunk pairs + metadata, **human-verified**
- `evals/golden/unanswerable.jsonl` — ~10 questions the corpus genuinely cannot answer (for abstention testing)
- `src/safety_rag/eval/metrics.py` — pure retrieval-metric functions, unit tested
- `scripts/run_eval.py` — runs the retrieval eval against the golden set, prints Hit@5, MRR@10, Recall@10
- `scripts/run_eval_gen.py` — runs the generation eval (LLM-as-judge faithfulness + citation accuracy + abstention rate)
- `.github/workflows/eval.yml` — runs the retrieval eval on every PR, fails if Hit@5 drops >2 points vs. main

## Prerequisites

- Phase 3 done (API in Docker, Qdrant running, `ask()` working, corpus re-indexed cleanly — no stale intermediate files in `data/processed/`)
- The pre-registered threshold from the README TL;DR: **Hit@5 ≥0.85** = vanilla RAG good enough, **<0.85** = proceed to Phase 5
- `MINIMAX_API_KEY` available locally and as a GitHub Actions secret

## Design decisions made up front

Two things the original plan left ambiguous. Deciding them now avoids rework later:

1. **Golden questions are filter-free.** A real user asking the API won't pass `regulation=`/`part=`/`article_num=` themselves — that's an internal implementation detail, not something the front-end exposes. So the golden set format does **not** carry those fields, and `run_eval.py` calls `ask(question, k=10)` with no filters. This measures retrieval quality under realistic conditions.

2. **Retrieve once at k=10, derive both Hit@5 and MRR@10/Recall@10 from the same list.** The original plan called `ask(question, k=5)` and then computed `mrr_at_10` from a 5-item list, which is not actually MRR@10 — it's MRR@5 mislabeled. Fetch 10 results per question; slice to the top 5 for Hit@5, use the full 10 for MRR@10 and Recall@10.

## Steps

### 1. Retrieval metrics (build and test this first)

Create `src/safety_rag/eval/metrics.py`:

```python
def hit_at_k(retrieved_ids: list[str], gold_id: str, k: int) -> int:
    return int(gold_id in retrieved_ids[:k])


def mrr_at_k(retrieved_ids: list[str], gold_id: str, k: int) -> float:
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid == gold_id:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return 1.0
    return len(gold_ids & set(retrieved_ids[:k])) / len(gold_ids)
```

These are pure functions — no network, no LLM — so write `tests/eval/test_metrics.py` covering: gold at position 0, gold at the last valid position, gold absent, empty `retrieved_ids`, `k` larger than the list, and `recall_at_k` with an empty `gold_ids` set. Get this file green before anything else in this phase — every later step depends on these three functions being correct, and they're the cheapest thing here to get wrong silently.

Golden set record format:
```json
{"q_id": "q042", "tier": "lepanto", "question": "…", "gold_ids": ["chunk_abc123"], "all_acceptable_ids": ["chunk_abc123", "chunk_def456"], "difficulty": "medium"}
```

`gold_ids` is the primary gold; `all_acceptable_ids` is the full set (for MRR/recall when multiple chunks legitimately answer). No `regulation`/`part`/`article_num` fields — per the design decision above, retrieval runs unfiltered.

### 2. Build the golden set generation prompt

Create `src/safety_rag/eval/golden_generator.py`. It takes a chunk, asks the LLM to produce:

```json
{
  "question": "A natural question this chunk answers",
  "difficulty": "easy" | "medium" | "hard",
  "answerable": true
}
```

Use MiniMax-M2 with **reasoning mode ON** and **temperature 0.2** for diverse-but-grounded generation. Sample ~3 questions per chunk, across the whole corpus (530 chunks) — cost isn't a constraint here, so generate broadly rather than sampling a subset; a larger candidate pool gives you more to choose from during vetting.

Check first whether `generation/llm.py`'s existing `generate()` supports a reasoning-mode toggle and can be made to return structured JSON reliably. If it currently only does plain chat completion, extend it (an optional `reasoning: bool` param, and either a JSON-mode API flag or a strict "respond with JSON only" system prompt plus a parse-and-retry loop) rather than duplicating a second client here.

### 3. Generate, then **manually vet** every question

Create `scripts/build_golden.py`:

```bash
uv run python scripts/build_golden.py --regulation ai_act --output evals/golden/ai_act.candidates.jsonl
uv run python scripts/build_golden.py --regulation nis2 --output evals/golden/nis2.candidates.jsonl
```

This generates candidate questions across the full corpus — expect several hundred. **Do not commit them yet.** Read every one and:

- Delete questions that are vague or unanswerable from the chunk
- Fix wording for clarity
- Add the source chunk's `content_hash` so the eval ties the question to a specific corpus version
- Tag with `tier: lepanto` or `tier: demo`
- Move vetted questions into `evals/golden/lepanto.jsonl` and `evals/golden/demo.jsonl`

**Time sink:** plan for 2–3+ hours of manual vetting, more if you generate a larger pool. This is the most important quality control in the project. A bad question poisons every future metric.

### 4. Build the unanswerable set

Hand-author 8–12 questions that the corpus genuinely cannot answer. Examples:
- "What is the deadline for AI Act enforcement under the Polish Data Protection Act?"
- "How does AI Act Article 26 interact with the GDPR Article 22 right to explanation?"
- "What is the procedure for NIS2 fines in the Netherlands?"

For each, the eval should expect the model to say "The corpus does not address this." Anything else is a failure.

Save to `evals/golden/unanswerable.jsonl`.

### 5. The retrieval eval runner

Create `scripts/run_eval.py`:

```python
import json
import logging
import statistics
from pathlib import Path

from safety_rag.api.ask import ask
from safety_rag.eval.metrics import hit_at_k, mrr_at_k, recall_at_k

logger = logging.getLogger(__name__)


def run_retrieval_eval(golden_path: Path, k_hit: int = 5, k_rank: int = 10) -> dict:
    hits, mrrs, recalls, failures = [], [], [], []

    for line in golden_path.read_text().splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        try:
            result = ask(q["question"], k=k_rank)
        except Exception:
            logger.exception("eval question failed: %s", q.get("q_id"))
            failures.append(q.get("q_id"))
            continue

        ids = [s["content_hash"] for s in result["sources"]]
        hits.append(hit_at_k(ids, q["gold_ids"][0], k_hit))
        mrrs.append(mrr_at_k(ids, q["gold_ids"][0], k_rank))
        recalls.append(recall_at_k(ids, set(q["all_acceptable_ids"]), k_rank))

    if failures:
        logger.warning("%d/%d questions failed and were excluded: %s",
                        len(failures), len(failures) + len(hits), failures)

    return {
        "hit_at_5": statistics.mean(hits) if hits else 0.0,
        "mrr_at_10": statistics.mean(mrrs) if mrrs else 0.0,
        "recall_at_10": statistics.mean(recalls) if recalls else 0.0,
        "n_questions": len(hits),
        "n_failed": len(failures),
    }
```

A failed question is logged and excluded rather than crashing the whole run — a single flaky network call or a malformed golden line shouldn't discard every other result. `n_failed` is reported in the output so a bad run is visible rather than silently averaging over fewer questions than intended; treat a nonzero `n_failed` as a signal to investigate before trusting the aggregate numbers, not something to ignore.

Run it:
```bash
uv run python scripts/run_eval.py --tier lepanto
uv run python scripts/run_eval.py --tier demo
```

Write results to `evals/results/{timestamp}.json` (gitignored).

### 6. Generation metrics (LLM-as-judge)

Create `src/safety_rag/eval/judge.py`:

```python
JUDGE_PROMPT = """You are an evaluator for a regulatory RAG system.

Question: {question}
Retrieved context:
{context}
Answer: {answer}

For each criterion, respond with JSON only:
- faithfulness: "yes" | "no" | "partial" — Is every claim in the answer supported by the retrieved context? Quote any unsupported span.
- citation_accuracy: "yes" | "no" — Does the cited Article actually contain the claim? Compare the cited text vs. the claim.
- abstention: "yes" | "no" — For an unanswerable question, did the answer correctly abstain?

JSON:"""
```

Use MiniMax-M2 with reasoning ON for this step. Parse the JSON output; if parsing fails, retry once with a stricter "output only valid JSON, no prose" reminder before giving up and logging the question as unjudged (same failure-tolerant pattern as step 5). Report per-question + aggregate.

### 7. The generation eval runner

`scripts/run_eval_gen.py` is similar to `run_eval.py` but calls `ask()`, feeds the answer + context to `judge()`, aggregates. Since cost isn't a constraint, this can run as often as you like rather than being reserved for nightly-only — but it's still slower than the retrieval eval (one LLM call per question for generation, plus one for judging), so budget wall-clock time accordingly, especially against the full unfiltered golden set.

```bash
uv run python scripts/run_eval_gen.py --tier lepanto
```

### 8. The CI gate

Create `.github/workflows/eval.yml`:

```yaml
name: eval
on:
  pull_request:
    branches: [main]
jobs:
  retrieval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12"}
      - run: pip install uv
      - run: uv sync --all-extras
      - run: docker run -d --rm -p 6333:6333 --name qdrant-ci qdrant/qdrant
      - name: Wait for Qdrant
        run: |
          for i in $(seq 1 30); do
            curl -sf http://localhost:6333/collections && break
            sleep 2
          done
      - run: uv run python scripts/build_index.py
      - name: Eval lepanto tier
        run: uv run python scripts/run_eval.py --tier lepanto --output eval_run_lepanto.json
      - name: Eval demo tier
        run: uv run python scripts/run_eval.py --tier demo --output eval_run_demo.json
      - name: Compare to main
        run: |
          uv run python scripts/eval_gate.py \
            --current eval_run_lepanto.json \
            --baseline-url https://raw.githubusercontent.com/joelsansana/eu-ai-rag/main/evals/results/baseline.json \
            --metric hit_at_5 \
            --tolerance 0.02
```

Two fixes from the original plan:

- **Polling instead of a fixed `sleep 10`.** CI runners vary in speed; a fixed sleep either wastes time or isn't long enough. Poll `/collections` until it responds, capped at ~60 seconds total.
- **Missing-baseline bootstrap.** The very first PR after this phase lands will run before `evals/results/baseline.json` exists on `main`, so the fetch will 404. `scripts/eval_gate.py` should treat a missing/unfetchable baseline as "nothing to compare against — pass," and log a warning, rather than failing the PR that's trying to create the baseline in the first place:

```python
try:
    baseline = fetch_baseline(args.baseline_url)
except BaselineNotFoundError:
    print("No baseline found on main yet — skipping gate, passing by default.")
    sys.exit(0)
```

`scripts/eval_gate.py` fetches `baseline.json` from main and compares — fails if `current_hit_at_5 < baseline_hit_at_5 - 0.02`.

Generate the baseline once locally and commit it:
```bash
uv run python scripts/run_eval.py --tier lepanto --output evals/results/baseline.json
git add evals/results/baseline.json
git commit -m "Phase 4: eval baseline"
git push origin main
```

### 9. Commit

```bash
git add src/safety_rag/eval scripts tests .github/workflows/eval.yml evals/golden evals/results/baseline.json
git commit -m "Phase 4: golden set + metrics + CI gate"
git push origin main
```

Open a test PR. **Verify:** CI runs the eval, prints Hit@5, gate passes on a no-change PR.

## Verify phase complete

- All three golden files exist and total ≥50 questions
- `run_eval.py` produces numbers (Hit@5, MRR@10, Recall@10) for both tiers, retrieved from a single unfiltered `k=10` call per question
- Numbers are sensible: ≥0.7 at minimum for vanilla RAG on this structured corpus
- CI gate works: artificially regressing Hit@5 by 3+ points in a test branch fails CI; a PR opened before any baseline exists passes rather than erroring
- The **pre-registered threshold** is met or not — either is fine, but the number goes into the README's results table

## Pitfalls

- **Golden set quality is everything.** A vague question gets vague retrieval. Spend real time vetting. Better 20 great questions than 60 mediocre ones, even with a larger candidate pool to choose from.
- **`reasoning_mode` on the judge.** Off, and you'll get yes/no answers without justification. On, and the judge quotes the relevant span.
- **Eval drift across commits.** The golden set + corpus hashes must be in lockstep. This project already hit a version of this bug in Phase 3 — a stale intermediate JSONL produced chunks with `content_hash=None`, silently duplicating points in the index. If you ever re-run the chunker or re-download the corpus and hashes change, the golden set's `gold_ids` go stale and every metric silently degrades without an obvious cause. Re-verify the golden set against the current index after any re-chunking.
- **Don't commit `evals/results/`.** These are timestamped, regeneratable, and clutter git history. `.gitignore` should cover this directory except `baseline.json`.
- **The 2-point tolerance is a heuristic.** If your eval is noisy (Hit@5 varies by ±1 point run-to-run), tighten by re-running the baseline multiple times and averaging. Don't set tolerance so loose you never catch regressions.
- **LLM-as-judge bias.** The judge is itself a model. Keep ~20% of the golden set aside for human judging later. If the LLM judge diverges from your judgment by >10% on that subset, flag it in the README's "Limitations" section.

## What's next

If Hit@5 ≥ 0.85: **vanilla RAG is good enough**. Skip Phase 5, write up the results, move to Phase 6 (README polish, demo, launch post).

If Hit@5 < 0.85: Phase 5 — Retrieval v2. Add BM25 + dense hybrid + a cross-encoder reranker, re-run the eval, quantify the improvement.
