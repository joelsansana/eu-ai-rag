# Phase 5 — Retrieval v2 (only if Phase 4 said vanilla RAG wasn't enough)

> Hybrid search (BM25 + dense) with reciprocal rank fusion, then a cross-encoder reranker. ~4–5 hours.

## When to do this phase

Run Phase 4 first. Look at the Hit@5 number for the `lepanto` tier (the harder one, with customer-shaped questions).

- **Hit@5 ≥ 0.85**: Skip this phase. Vanilla RAG is good enough for the structured regulatory corpus. The story you tell in the README is "structured corpus + contextual chunk headers + MiniMax-M2 generation = vanilla RAG is fine." Move to Phase 6.
- **Hit@5 < 0.85**: Do this phase. Add hybrid + rerank, re-evaluate, publish v1 → v2 numbers.

This phase is **conditional**, not required. Either outcome is a valid story.

## Goal

Add two retrieval layers on top of Phase 2's dense baseline:
1. **BM25 sparse retrieval** over the same chunked text
2. **Reciprocal rank fusion (RRF)** combining BM25 and dense top-k
3. **Cross-encoder reranker** (`BAAI/bge-reranker-base`) on the fused top-20

Then re-run Phase 4's eval and update the README with v1 vs v2 numbers.

## Prerequisites

- Phase 4 done with Hit@5 < 0.85
- Two new dependencies: `rank_bm25`, the bge-reranker-base model (downloaded automatically by sentence-transformers)

```bash
uv add rank_bm25
```

## Steps

### 1. Build the BM25 index

Create `src/safety_rag/retrieval/bm25.py`. BM25 is cheap and index-friendly because the corpus is small (~600 chunks):

```python
from rank_bm25 import BM25Okapi
import json, re
from pathlib import Path

class BM25Index:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        tokenized = [self._tokenize(c["header"] + "\n" + c["text"]) for c in chunks]
        self.index = BM25Okapi(tokenized)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        # simple whitespace + lower; the corpus is small and well-formed
        return re.findall(r"\w+", text.lower())

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        tokens = self._tokenize(query)
        scores = self.index.get_scores(tokens)
        return sorted(enumerate(scores), key=lambda x: -x[1])[:k]
```

Build the index once at startup (or in `scripts/build_index.py`):
```python
from safety_rag.retrieval.bm25 import BM25Index
chunks = [...]  # load from data/processed/*.jsonl
bm25 = BM25Index(chunks)
```

### 2. Reciprocal rank fusion

In `src/safety_rag/retrieval/hybrid.py`:

```python
def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal rank fusion. rankings is a list of ranked-id lists.
    Returns the fused ranking."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)
```

### 3. Cross-encoder reranker

In `src/safety_rag/retrieval/reranker.py`:

```python
from sentence_transformers import CrossEncoder

_MODEL = None

def get_model() -> CrossEncoder:
    global _MODEL
    if _MODEL is None:
        _MODEL = CrossEncoder("BAAI/bge-reranker-base")
    return _MODEL

def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Returns top_k chunks sorted by cross-encoder score."""
    model = get_model()
    pairs = [(query, c["header"] + "\n" + c["text"]) for c in chunks]
    scores = model.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: -x[1])
    return [c for c, _ in ranked[:top_k]]
```

### 4. Hybrid search function

In the same `hybrid.py`:

```python
from safety_rag.retrieval.vector_store import search as dense_search
from safety_rag.retrieval.bm25 import BM25Index
from safety_rag.retrieval.embedder import embed
from safety_rag.retrieval.reranker import rerank

def hybrid_search(question: str, bm25: BM25Index, *,
                  dense_k: int = 20, bm25_k: int = 20, rerank_k: int = 5,
                  **filters) -> list[dict]:
    # Dense
    q_emb = embed([question])[0]
    dense_results = dense_search(q_emb, k=dense_k, **filters)
    dense_ranking = [r["payload"]["content_hash"] for r in dense_results]
    dense_lookup = {r["payload"]["content_hash"]: r for r in dense_results}

    # BM25
    bm25_results = bm25.search(question, k=bm25_k)
    bm25_ranking = [bm25.chunks[i]["content_hash"] for i, _ in bm25_results]
    bm25_lookup = {bm25.chunks[i]["content_hash"]: bm25.chunks[i]
                   for i, _ in bm25_results}

    # RRF
    fused_ids = rrf([dense_ranking, bm25_ranking])

    # Hydrate to chunk dicts (preferring dense payload for metadata)
    candidates = []
    for cid in fused_ids:
        if cid in dense_lookup:
            candidates.append(dense_lookup[cid])
        elif cid in bm25_lookup:
            candidates.append(bm25_lookup[cid])

    # Rerank
    return rerank(question, candidates, top_k=rerank_k)
```

### 5. Wire into `ask()`

Update `src/safety_rag/api/ask.py` to use `hybrid_search` when `use_hybrid=True`. Default to `False` (v1 path) so you can A/B compare in the eval.

### 6. Re-run Phase 4's eval

```bash
# v1 (already have this number from Phase 4)
uv run python scripts/run_eval.py --tier lepanto --k 5 --output evals/results/v1_lepanto.json

# v2
uv run python scripts/run_eval.py --tier lepanto --k 5 --use_hybrid --output evals/results/v2_lepanto.json
```

Compare:
```bash
uv run python scripts/eval_diff.py v1_lepanto.json v2_lepanto.json
```

Expected delta: +0.03 to +0.08 Hit@5. If the delta is <0.03, the cross-encoder didn't help — keep hybrid but drop the reranker. If the delta is ≥0.05, both layers earn their keep.

### 7. Update the README

Add a results table:

```
## Results (Phase 5, 2026-MM-DD)

| Tier      | Phase | Hit@5 | MRR@10 | Recall@10 | Δ Hit@5 |
|-----------|-------|-------|--------|-----------|---------|
| lepanto   | v1    | 0.82  | 0.74   | 0.88      |   —     |
| lepanto   | v2    | 0.89  | 0.83   | 0.94      | +0.07   |
| demo      | v1    | 0.91  | 0.85   | 0.95      |   —     |
| demo      | v2    | 0.93  | 0.88   | 0.96      | +0.02   |
```

If the delta is small, **say so**. The honest story ("structured corpus + contextual headers + naive RAG is already strong; cross-encoder helps 3 points on harder queries") is more defensible than overselling.

### 8. Commit

```bash
git add src/safety_rag/retrieval scripts
git commit -m "Phase 5: hybrid search + cross-encoder reranker"
git push origin main
```

## Verify phase complete

- v2 eval numbers exist for both tiers
- README has a results table with v1 vs v2
- CI gate still works (rerun on a test PR to confirm)
- The story is honest (no cherry-picked metrics)

## Pitfalls

- **BM25 tokenization matters.** The default in `rank_bm25` is whitespace; that's fine for English regulatory text but useless for multi-language. If you ever add DE/FR, switch to a proper tokenizer.
- **RRF k parameter.** The classic value is k=60. Don't tune this on the golden set — it overfits fast.
- **Reranker latency.** `bge-reranker-base` is ~280M params. On CPU, expect ~200ms per query for top-20. Move to GPU if `/ask` latency becomes a complaint.
- **Don't compare v1 vs v2 on different golden sets.** Same golden set, same corpus hash, same evaluation date. Otherwise the number is noise.
- **The pre-registered threshold matters more than the v2 lift.** A v1 → v2 lift of 0.07 is nice. But the headline is "did v1 already meet the threshold?" If yes, the lift is a footnote.

## What's next

Phase 6 — Story layer (README centerpiece, Mermaid diagram, demo GIF, LinkedIn post). The code is done; now you sell it.
