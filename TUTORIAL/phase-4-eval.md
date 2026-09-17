# Phase 4 — Evaluation (the reason this project exists)

> Golden set + retrieval metrics + generation metrics (LLM-as-judge) + CI regression gate. ~8–10 hours.

## Goal

- `evals/golden/lepanto.jsonl` — ~25 customer-shaped Q→source-chunk pairs + metadata, **human-verified**
- `evals/golden/demo.jsonl` — ~25 regulation-shaped Q→source-chunk pairs + metadata, **human-verified**
- `evals/golden/unanswerable.jsonl` — ~10 questions the corpus genuinely cannot answer (for abstention testing)
- `scripts/run_eval.py` — runs the retrieval eval against the golden set, prints Hit@5, MRR@10, Recall@10
- `scripts/run_eval_gen.py` — runs the generation eval (LLM-as-judge faithfulness + citation accuracy + abstention rate), nightly via repo secret
- `.github/workflows/eval.yml` — runs the retrieval eval on every PR, fails if Hit@5 drops >2 points vs. main

## Prerequisites

- Phase 3 done (API in Docker, Qdrant running, `ask()` working)
- The pre-registered threshold from the README TL;DR: **Hit@5 ≥0.85** = vanilla RAG good enough, **<0.85** = proceed to Phase 5
- `MINIMAX_API_KEY` as a GitHub Actions secret for the nightly generation eval

## Steps

### 1. Build the golden set generation prompt

Create `src/safety_rag/eval/golden_generator.py`. It takes a chunk, asks the LLM to produce:

```json
{
  "question": "A natural question this chunk answers",
  "difficulty": "easy" | "medium" | "hard",
  "answerable": true
}
```

Use MiniMax-M2 with **reasoning mode ON** and **temperature 0.2** for diverse-but-grounded generation. Sample ~3 questions per chunk.

### 2. Generate, then **manually vet** every question

Create `scripts/build_golden.py`:

```bash
uv run python scripts/build_golden.py --regulation ai_act --output evals/golden/ai_act.candidates.jsonl
uv run python scripts/build_golden.py --regulation nis2 --output evals/golden/nis2.candidates.jsonl
```

This generates ~150–300 candidate questions. **Do not commit them yet.** Read every one and:

- Delete questions that are vague or unanswerable from the chunk
- Fix wording for clarity
- Add the source chunk's `content_hash` so the eval ties the question to a specific corpus version
- Tag with `tier: lepanto` or `tier: demo`
- Move vetted questions into `evals/golden/lepanto.jsonl` and `evals/golden/demo.jsonl`

**Time sink:** plan for 2–3 hours of manual vetting. This is the most important quality control in the project. A bad question poisons every future metric.

### 3. Build the unanswerable set

Hand-author 8–12 questions that the corpus genuinely cannot answer. Examples:
- "What is the deadline for AI Act enforcement under the Polish Data Protection Act?"
- "How does AI Act Article 26 interact with the GDPR Article 22 right to explanation?"
- "What is the procedure for NIS2 fines in the Netherlands?"

For each, the eval should expect the model to say "The corpus does not address this." Anything else is a failure.

Save to `evals/golden/unanswerable.jsonl`.

### 4. Retrieval metrics

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

Golden set record format:
```json
{"q_id": "q042", "tier": "lepanto", "question": "…", "gold_ids": ["chunk_abc123"], "all_acceptable_ids": ["chunk_abc123", "chunk_def456"], "difficulty": "medium"}
```

`gold_ids` is the primary gold; `all_acceptable_ids` is the full set (for MRR/recall when multiple chunks legitimately answer).

### 5. The retrieval eval runner

Create `scripts/run_eval.py`:

```python
import json, statistics
from pathlib import Path
from safety_rag.api.ask import ask
from safety_rag.eval.metrics import hit_at_k, mrr_at_k, recall_at_k


def run_retrieval_eval(golden_path: Path, k: int = 5) -> dict:
    hits, mrrs, recalls = [], [], []
    for line in golden_path.read_text().splitlines():
        q = json.loads(line)
        result = ask(q["question"], k=k)
        ids = [s["content_hash"] for s in result["sources"]]
        hits.append(hit_at_k(ids, q["gold_ids"][0], k))
        mrrs.append(mrr_at_k(ids, q["gold_ids"][0], k))
        recalls.append(recall_at_k(ids, set(q["all_acceptable_ids"]), k))
    return {
        "hit_at_5": statistics.mean(hits),
        "mrr_at_10": statistics.mean(mrrs),
        "recall_at_10": statistics.mean(recalls),
        "n_questions": len(hits),
    }
```

Run it:
```bash
uv run python scripts/run_eval.py --tier lepanto --k 5
uv run python scripts/run_eval.py --tier demo --k 5
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

Use MiniMax-M2 with reasoning ON for this step. Parse the JSON output. Report per-question + aggregate.

### 7. The generation eval runner

`scripts/run_eval_gen.py` is similar to `run_eval.py` but calls `ask()`, feeds the answer + context to `judge()`, aggregates. This one runs **manually/nightly** — it's slower (one LLM call per question for generation, plus one for judging) and costs ~€0.50 per run.

```bash
MINIMAX_API_KEY=*** uv run python scripts/run_eval_gen.py --tier lepanto
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
      - run: sleep 10
      - run: uv run python scripts/build_index.py
      - name: Eval lepanto tier
        run: uv run python scripts/run_eval.py --tier lepanto --k 5 --output eval_run.json
      - name: Eval demo tier
        run: uv run python scripts/run_eval.py --tier demo --k 5 --output eval_run.json
      - name: Compare to main
        run: |
          uv run python scripts/eval_gate.py \
            --current eval_run.json \
            --baseline https://raw.githubusercontent.com/joelsansana/eu-ai-rag/main/evals/results/baseline.json \
            --metric hit_at_5 \
            --tolerance 0.02
```

`scripts/eval_gate.py` fetches `baseline.json` from main and compares — fails if `current_hit_at_5 < baseline_hit_at_5 - 0.02`.

Generate the baseline once locally and commit it:
```bash
uv run python scripts/run_eval.py --tier lepanto --output evals/results/baseline.json
cp evals/results/baseline.json evals/results/baseline.json  # canonical path
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
- `run_eval.py` produces numbers (Hit@5, MRR, Recall) for both tiers
- Numbers are sensible: ≥0.7 at minimum for vanilla RAG on this structured corpus
- CI gate works: artificially regressing Hit@5 by 3+ points in a test branch fails CI
- The **pre-registered threshold** is met or not — either is fine, but the number goes into the README's results table

## Pitfalls

- **Golden set quality is everything.** A vague question gets vague retrieval. Spend 2–3 hours vetting. Better 20 great questions than 60 mediocre ones.
- **`reasoning_mode` on the judge.** Off, and you'll get yes/no answers without justification. On, and the judge quotes the relevant span.
- **Eval drift across commits.** The golden set + corpus hashes must be in lockstep. If you re-download the corpus (different hash), you need to re-verify the golden set.
- **Don't commit `evals/results/`.** These are timestamped, regeneratable, and clutter git history. `.gitignore` should cover this directory except `baseline.json`.
- **The 2-point tolerance is a heuristic.** If your eval is noisy (Hit@5 varies by ±1 point run-to-run), tighten by re-running the baseline multiple times and averaging. Don't set tolerance so loose you never catch regressions.
- **LLM-as-judge bias.** The judge is itself a model. Keep ~20% of the golden set aside for human judging later. If the LLM judge diverges from your judgment by >10% on that subset, flag it in the README's "Limitations" section.

## What's next

If Hit@5 ≥ 0.85: **vanilla RAG is good enough**. Skip Phase 5, write up the results, move to Phase 6 (README polish, demo, launch post).

If Hit@5 < 0.85: Phase 5 — Retrieval v2. Add BM25 + dense hybrid + a cross-encoder reranker, re-run the eval, quantify the improvement.
