# Phase 2 — Retrieval v1 (deliberately naive)

> Dense retrieval + naive RAG answering with citations. ~4–5 hours.

## Goal

A `safety_rag.ask(question: str) -> Answer` function that:
1. Embeds the question with `bge-small-en-v1.5`
2. Top-k=5 dense search over Qdrant, with optional metadata filters
3. Builds a prompt with the retrieved chunks and the question
4. Calls MiniMax-M2 (reasoning off, streaming off) to generate an answer
5. Returns the answer **plus cited source chunks** (regulation, article_num, recital_num, text, score)

Plus a CLI smoke test you can run by hand.

## Prerequisites

- Phase 1 done (chunks in `data/processed/*.jsonl`)
- Docker running with a Qdrant container (start: `docker run -p 6333:6333 qdrant/qdrant`)
- `MINIMAX_API_KEY` set in your shell (get one from the MiniMax console)
- No new dependencies beyond `sentence-transformers`, `qdrant-client`, `httpx`, `openai` (the OpenAI client pointed at MiniMax is `openai>=1.0`)

```bash
uv add sentence-transformers qdrant-client httpx openai
```

## Steps

### 1. Configure the MiniMax client

Create `src/safety_rag/generation/llm.py`. The OpenAI-compatible client pointed at MiniMax:

```python
from openai import OpenAI
import os


def get_client() -> OpenAI:
    return OpenAI(
        base_url="https://api.minimax.io/anthropic/v1",
        api_key=os.environ["MINIMAX_API_KEY"],
    )


def generate(prompt: str, *, reasoning: bool = False) -> str:
    client = get_client()
    resp = client.chat.completions.create(
        model="MiniMax-M2",
        messages=[{"role": "user", "content": prompt}],
        reasoning={"effort": "medium"} if reasoning else None,
    )
    return resp.choices[0].message.content
```

**Verify by smoke test:**
```bash
MINIMAX_API_KEY=*** uv run python -c "
from safety_rag.generation.llm import generate
print(generate('Reply with one word: pong.'))
"
```

Should print `pong` (or similar one-word answer). If you get an auth error, double-check the key and the `base_url`.

### 2. Build the embedder

Create `src/safety_rag/retrieval/embedder.py`:

```python
from sentence_transformers import SentenceTransformer

_MODEL = None


def get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _MODEL


def embed(texts: list[str]) -> list[list[float]]:
    return get_model().encode(texts, normalize_embeddings=True).tolist()
```

`normalize_embeddings=True` is critical — Qdrant uses cosine similarity and normalized vectors make the inner product equivalent.

**Verify:**
```bash
uv run python -c "
from safety_rag.retrieval.embedder import embed
v = embed(['test sentence'])
print(len(v), len(v[0]))
"
```

Should print `1 384` (bge-small-en-v1.5 has 384 dims).

### 3. Build the Qdrant store

Create `src/safety_rag/retrieval/vector_store.py`. Two operations: `upsert_chunks(chunks)` and `search(query_embedding, *, regulation=None, article_num=None, k=5)`.

Key choices:
- Distance: `COSINE`
- Collection name: `safety_rag_v1`
- Payload fields: `regulation`, `part`, `article_num`, `recital_num`, `annex_id`, `chapter`, `celex`, `header`, `text`, `content_hash`
- Use `qdrant_client.QdrantClient(url="http://localhost:6333")` for the local dev case

Write a CLI `scripts/build_index.py` step that:
1. Loads `data/processed/*.jsonl`
2. Embeds in batches of 64 (Qdrant's `batch_size`)
3. Upserts to Qdrant
4. Reports total points

```bash
docker run -d --rm -p 6333:6333 --name qdrant-dev qdrant/qdrant
uv run python scripts/build_index.py
```

**Verify:** `curl http://localhost:6333/collections/safety_rag_v1 | jq` shows `"vectors_count"` around 500–800 (sum of both JSONLs).

### 4. Write the retrieval function

In the same `vector_store.py`:

```python
def search(
    query_embedding: list[float], *, k: int = 5, **filters
) -> list[dict]:
    """Returns [{'id': str, 'score': float, 'payload': {...}}, ...] sorted by score desc."""
```

For filters, use `models.Filter` with `must` conditions on payload fields. Examples: `regulation="ai_act"`, `article_num=26`, `part="recital"`.

### 5. Write the prompt template

Create `src/safety_rag/generation/prompts.py`:

```python
RAG_PROMPT = """You are a regulatory compliance assistant for industrial AI systems subject to EU regulation.

Answer the user's question using ONLY the retrieved context below. Cite every claim with the regulation, Article number (and chapter / annex if relevant), and quote the exact supporting text inline.

If the context does not contain enough information to answer, reply exactly: "The corpus does not address this."

Retrieved context:
{context}

User question: {question}

Answer (with inline citations in the form [Regulation X, Article Y]):"""
```

`{context}` will be the concatenation of:
```
[Source 1 | Regulation: ai_act | Article: 26 | Chapter: 4 — Deployer obligations]
<text of chunk 1>

[Source 2 | Regulation: nis2 | Article: 21 | Chapter: 4 — Cybersecurity risk-management measures]
<text of chunk 2>
…
```

Sort sources by score (best first). Truncate each chunk to 1500 chars to keep prompts manageable.

### 6. Write the end-to-end `ask` function

Create `src/safety_rag/api/ask.py` (or `src/safety_rag/retrieval/rag.py`, your call):

```python
def ask(question: str, *, k: int = 5, **filters) -> dict:
    q_emb = embed([question])[0]
    results = search(q_emb, k=k, **filters)
    context = format_context(results)
    answer = generate(RAG_PROMPT.format(context=context, question=question))
    return {
        "answer": answer,
        "sources": [{"score": r["score"], **r["payload"]} for r in results],
    }
```

### 7. Smoke test from the CLI

```bash
uv run python -c "
from safety_rag.api.ask import ask
r = ask('What are the deployer obligations under Article 26 of the AI Act?')
print('ANSWER:', r['answer'])
print()
print('SOURCES:')
for s in r['sources']:
    print(f'  [{s[\"score\"]:.3f}] {s[\"regulation\"]} Art. {s[\"article_num\"]} — {s[\"header\"]}')
"
```

**Verify:**
- Answer cites Article 26 explicitly
- Top source is Article 26 of the AI Act (score > 0.5)
- At least 3 of 5 sources are from the AI Act

### 8. Commit

```bash
git add src/safety_rag scripts
git commit -m "Phase 2: retrieval v1 (dense + naive RAG)"
git push origin main
```

## Verify phase complete

- Qdrant collection `safety_rag_v1` has 500–800 points
- `ask()` returns an answer with inline citations
- Manual smoke test on 5 questions shows relevant top-k hits
- CI green

## Pitfalls

- **Embedding batch size.** `model.encode(texts, batch_size=64)` is a sweet spot; too small is slow, too large OOMs on CPU.
- **Cosine vs dot-product.** With `normalize_embeddings=True`, both work. Pick one and stick to it. The vector store wrapper should hide this.
- **Prompt token budget.** With k=5 chunks at ~600 tokens each = 3000 tokens just for context. MiniMax-M2's 200K context is fine, but watch latency. Truncate chunk text in `format_context`.
- **`MINIMAX_API_KEY` in tests.** Don't commit it. Use a fixture key in tests; mock the generation step.
- **Reasoning mode.** Default off for the user-facing path (faster, cheaper). On for the LLM-as-judge in Phase 4. Don't mix them up.

## What's next

Phase 3 — API. You'll wrap `ask()` in a FastAPI app with `/search`, `/ask`, `/health`, `/stats`, `/golden`, `/feedback` endpoints, all in a Docker container with Qdrant.
