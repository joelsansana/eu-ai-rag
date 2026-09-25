# Phase 2 — Retrieval v1 (deliberately naive)

> Dense retrieval + naive RAG answering with citations. ~4–5 hours.
>
> **Status (verified 2026-09-25):** this guide was corrected against the actual code after the build was completed. The material deltas: (a) the OpenAI-compatible base URL is `https://api.minimax.io/v1` (the tutorial originally had `.../anthropic/v1`, which is the wrong path); (b) `generate(prompt)` is a plain function with **no `reasoning` parameter** — the tutorial's `reasoning={"effort": "medium"}` is not in the code; (c) the LLM response is post-processed by `_strip_thinking`, which removes `` blocks the model may emit inline; (d) `vector_store.search` accepts `regulation` and `article_num` as filter kwargs (forwarded via `ask(**filters)`); point IDs are deterministic via `uuid.uuid5(NAMESPACE_URL, content_hash)`; (e) `scripts/build_index.py` only loads pre-built JSONL and upserts to Qdrant — it does **not** parse or chunk; (f) the prompt assembly lives in `generation/prompts.py` as `build_prompt` and `format_context` helpers (the tutorial showed the prompt string concatenated inline).

## Goal

A `safety_rag.api.ask.ask(question: str, *, k: int = 5, **filters) -> dict` function that:

1. Embeds the question with `bge-small-en-v1.5`
2. Top-k=5 dense search over Qdrant, with optional metadata filters (`regulation`, `article_num`)
3. Builds the prompt from the retrieved chunks (sorted by score, truncated per chunk)
4. Calls `MiniMax-M2` (no reasoning mode) to generate an answer, with `` blocks stripped
5. Returns the answer **plus cited source chunks** (`score` + the full `payload`)

Plus a CLI smoke test you can run by hand (covered by `tests/api/test_ask.py::TestAskIntegration::test_ask_real_pipeline`, opt-in with `uv run pytest -m integration`).

## Prerequisites

- Phase 1 done (chunks in `data/processed/*.chunks.jsonl`)
- Docker running with a Qdrant container (`docker run -d --rm -p 6333:6333 --name qdrant-dev qdrant/qdrant`)
- `MINIMAX_API_KEY` set in your shell (get one from the MiniMax console)
- No new dependencies beyond `sentence-transformers`, `qdrant-client`, `httpx`, `openai` (already in `pyproject.toml` from Phase 0)

```bash
uv sync --all-extras
```

## Steps

### 1. Configure the MiniMax client

Create `src/safety_rag/generation/llm.py`. The OpenAI-compatible client pointed at MiniMax. **Note: the `base_url` is `https://api.minimax.io/v1` (no `/anthropic/` segment).** The `OpenAI` Python client works against any OpenAI-compatible endpoint by changing only the `base_url` and `api_key`.

```python
from openai import OpenAI
import os


def get_client() -> OpenAI:
    return OpenAI(
        base_url="https://api.minimax.io/v1",
        api_key=os.environ["MINIMAX_API_KEY"],
    )
```

`generate(prompt: str)` is intentionally **a single-argument function** — no `reasoning` flag, no `temperature` knob. The original tutorial sketch had a `reasoning={"effort": "medium"}` parameter; we removed it because (a) the user-facing path doesn't need it (faster, cheaper), and (b) the LLM-as-judge in Phase 4 will need reasoning on, so introducing the toggle now would mean remembering to flip it. Cleaner to add a second function (`generate_reasoning`) in Phase 4.

```python
def generate(prompt: str) -> str:
    client = get_client()

    response = client.chat.completions.create(
        model="MiniMax-M2",
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content or ""

    return _strip_thinking(raw)
```

`_strip_thinking` is a safety net — MiniMax may emit `` blocks inline in the response content even with thinking disabled. The regex strips them:

```python
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Remove any <think>...</think> reasoning block a model may emit
    inline in its response content (safety net even with thinking disabled)."""
    return _THINK_TAG_RE.sub("", text).strip()
```

**Verify by smoke test** (also covered by the integration test `tests/generation/test_llm.py::test_generate_pong`):

```bash
MINIMAX_API_KEY=*** uv run python -c "
from safety_rag.generation.llm import generate
print(repr(generate('Reply with one word: pong.')))
"
```

Should print `'pong'` (or similar one-word answer, after `_strip_thinking` runs). If you get an auth error, double-check the key and the `base_url`.

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

`normalize_embeddings=True` is critical — Qdrant uses cosine similarity, and normalized vectors make the inner product equivalent (so we don't need to normalise at query time either). The module-level `_MODEL` is a lazy singleton: tests reset it in `setup_method` to prevent state leakage.

**Verify:**

```bash
uv run python -c "
from safety_rag.retrieval.embedder import embed
v = embed(['test sentence'])
print(len(v), len(v[0]))
"
```

Should print `1 384` (bge-small-en-v1.5 has 384 dims). The integration test `TestEmbedIntegration::test_embed_real_model_shape_and_normalization` asserts the same: shape `(384,)` and unit length (`np.linalg.norm(vec) ≈ 1.0`).

### 3. Build the Qdrant store

Create `src/safety_rag/retrieval/vector_store.py`. The module exposes the constants and operations below.

```python
COLLECTION_NAME = "safety_rag_v1"
VECTOR_SIZE = 384  # BAAI/bge-small-en-v1.5 output dimension

PAYLOAD_FIELDS = (
    "regulation",
    "part",
    "article_num",
    "recital_num",
    "annex_id",
    "chapter",
    "celex",
    "header",
    "text",
    "content_hash",
)


def get_client(url: str = "http://localhost:6333") -> QdrantClient:
    """Local dev Qdrant client. Point at a different url for other envs."""
    return QdrantClient(url=url)


def ensure_collection(client, collection_name=COLLECTION_NAME) -> None:
    """Create the collection if it doesn't exist yet. Safe to call repeatedly."""
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=VECTOR_SIZE, distance=qmodels.Distance.COSINE
            ),
        )


def _point_id(chunk: dict) -> str:
    """Deterministic point id from content_hash, so re-running upsert on the
    same chunk updates it in place instead of creating a duplicate point.
    Falls back to a random id if content_hash is missing."""
    content_hash = chunk.get("content_hash")
    if content_hash:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, content_hash))
    return str(uuid.uuid4())


def _payload(chunk: dict) -> dict:
    """Restrict the upserted payload to the known fields."""
    return {field: chunk.get(field) for field in PAYLOAD_FIELDS}


def upsert_chunks(
    chunks, *, client=None, collection_name=COLLECTION_NAME
) -> int:
    """Embed chunk['text'] for each chunk and upsert into Qdrant.
    Returns the number of points upserted."""
    if not chunks:
        return 0
    client = client or get_client()
    ensure_collection(client, collection_name)

    texts = [chunk["text"] for chunk in chunks]
    vectors = embed(texts)

    points = [
        qmodels.PointStruct(
            id=_point_id(chunk), vector=vector, payload=_payload(chunk)
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    client.upsert(collection_name=collection_name, points=points)
    return len(points)


def search(
    query_embedding,
    *,
    regulation=None,
    article_num=None,
    k=5,
    client=None,
    collection_name=COLLECTION_NAME,
) -> list:
    """Cosine-similarity search, optionally filtered by regulation and/or
    article_num. Returns up to k ScoredPoint results, highest score first."""
    client = client or get_client()
    query_filter = _build_filter(regulation, article_num)

    response = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        query_filter=query_filter,
        limit=k,
    )
    return response.points
```

Key choices:

- **Distance:** `COSINE` (matches `normalize_embeddings=True`).
- **Collection name:** `safety_rag_v1`. The version suffix is so we can swap to `safety_rag_v2` if Phase 5's hybrid search needs a fresh collection without a long migration.
- **Payload fields:** the explicit `PAYLOAD_FIELDS` tuple. `_payload` filters every chunk to these fields — extra fields in the chunk dict are silently dropped.
- **Point IDs:** deterministic via `uuid.uuid5(NAMESPACE_URL, content_hash)`. This means re-running `upsert_chunks` on the same chunk updates the point in place rather than creating a duplicate. Falls back to `uuid.uuid4()` if `content_hash` is missing.
- **Filters:** `regulation` and `article_num` are the only two filters wired up (more can be added by extending `_build_filter`).

### 4. Build the index from chunks JSONL

`scripts/build_index.py` loads chunks from `data/processed/*.jsonl` and upserts them in batches. It does **not** parse or chunk — that's Phase 1's job. The script's responsibilities are:

1. Find every `*.jsonl` file under `data/processed/` (sorted)
2. Stream chunks line-by-line
3. Batch them (default batch size 64)
4. For each batch, call `upsert_chunks(batch, client=client)` (which embeds + upserts)
5. Report progress per batch + final collection point count via `client.count(...).count`

```bash
docker run -d --rm -p 6333:6333 --name qdrant-dev qdrant/qdrant
uv run python scripts/build_index.py
```

Customise via flags:

```bash
uv run python scripts/build_index.py --data-dir data/processed --batch-size 64
```

**Verify:** the script prints `Done. Points upserted this run: <N>` and `Total points in 'safety_rag_v1': <N>`. The exact total depends on the corpus; expect ~600–800 for AI Act + NIS2 combined.

### 5. Build the prompt template

Create `src/safety_rag/generation/prompts.py`. The module exposes the prompt constant and three helpers (`_truncate`, `_format_source_header`, `format_context`, `build_prompt`).

```python
RAG_PROMPT = """You are a regulatory compliance assistant for industrial AI
systems subject to EU regulation.

Answer the user's question using ONLY the retrieved context below. Cite every
claim with the regulation, Article number (and chapter / annex if relevant),
and quote the exact supporting text inline.

If the context does not contain enough information to answer, reply exactly:
"The corpus does not address this."

Retrieved context:
{context}

User question: {question}

Answer (with inline citations in the form [Regulation X, Article Y]):"""

MAX_CHUNK_CHARS = 1500
```

The `{context}` placeholder is filled by `format_context`, which:

- Sorts results by `.score` descending
- For each result, builds a `[Source N | Regulation: … | Article: … | Chapter: … | Annex: … | Recital: …]` header (only including fields that are present)
- Truncates each chunk's text to `MAX_CHUNK_CHARS` (1500) with `_truncate` (which adds `…` if it cuts)
- Joins blocks with blank lines

```python
def build_prompt(question: str, results) -> str:
    """Assemble the full RAG_PROMPT with retrieved context and the question."""
    context = format_context(results)
    return RAG_PROMPT.format(context=context, question=question)
```

**Important:** the prompt format uses `RAG_PROMPT.format(...)`, which raises `KeyError` if the chunk text contains literal `{` / `}` (e.g. `Annex {III}` in regulatory text). The test suite explicitly covers this with `test_context_with_curly_braces_does_not_break_formatting`. **No fix needed in the current code** — the tests pass — but be aware of the constraint if you ever change how the prompt is assembled.

### 6. Write the end-to-end `ask` function

Create `src/safety_rag/api/ask.py`. It chains embed → search → build_prompt → generate and returns both the answer and the source payloads.

```python
def ask(question: str, *, k: int = 5, **filters) -> dict:
    """End-to-end RAG: embed the question, retrieve the top-k matching
    chunks, generate an answer grounded in that context, and return both.

    `filters` are forwarded to vector_store.search (currently `regulation`
    and/or `article_num`), e.g. ask("...", regulation="ai_act").
    """
    q_emb = embed([question])[0]
    results = search(q_emb, k=k, **filters)

    prompt = build_prompt(question, results)
    answer = generate(prompt)

    return {
        "answer": answer,
        "sources": [{"score": r.score, **(r.payload or {})} for r in results],
    }
```

Note: `sources` is a list of dicts merging `score` with the full `payload`. A result with `payload=None` yields `{"score": r.score}` only — the `or {}` defensive default.

### 7. Smoke test from the CLI

The integration test `tests/api/test_ask.py::TestAskIntegration::test_ask_real_pipeline` covers this end-to-end. Run it with:

```bash
uv run pytest -m integration tests/api/test_ask.py -v
```

The test asserts:

- `r["answer"]` is a non-empty string
- `r["sources"]` is a non-empty list
- Every source has a float `score` and the keys `regulation`, `article_num`, `header`

The original tutorial said "the top source should be Article 26 of the AI Act with score > 0.5" — the actual integration test is **softer**: it logs a NOTE if the top source isn't from `ai_act`, but doesn't fail. This is because the exact ranking depends on the corpus version Cellar serves, and we want the test to be robust against benign retrieval drift.

### 8. Commit

```bash
git add src/safety_rag scripts/build_index.py tests
git commit -m "Phase 2: retrieval v1 (dense + naive RAG)"
git push origin main
```

## Verify phase complete

- Qdrant collection `safety_rag_v1` has 600–800 points
- `uv run pytest -q` passes (smoke test + ingestion tests + chunker edge cases + retrieval tests + prompt tests + ask tests)
- `uv run pytest -m integration tests/api/test_ask.py -v` (with Qdrant running and `MINIMAX_API_KEY` set) passes the end-to-end pipeline
- CI green

## Pitfalls

- **Embedding batch size.** `model.encode(texts, batch_size=64)` is a sweet spot; too small is slow, too large OOMs on CPU. `scripts/build_index.py` uses 64 by default — override with `--batch-size` if you hit memory issues.
- **Cosine vs dot-product.** With `normalize_embeddings=True`, both work. Pick one and stick to it. The vector store wrapper hides this — `Distance.COSINE` in `ensure_collection` and `normalize_embeddings=True` in `embed` together make it correct.
- **Prompt token budget.** With k=5 chunks at ~600 tokens each = 3000 tokens just for context. MiniMax-M2's 200K context is fine, but watch latency. `MAX_CHUNK_CHARS = 1500` truncates each chunk to ~375 tokens; if you need more context per chunk, bump it and re-run the latency smoke test.
- **`MINIMAX_API_KEY` in tests.** Don't commit it. The unit tests mock the OpenAI client via `unittest.mock.patch`; integration tests need a real key and skip by default. CI doesn't run integration tests.
- **Reasoning mode is not wired.** `generate()` doesn't take a `reasoning` flag. If you need reasoning (e.g. for the Phase 4 LLM-as-judge), add a separate `generate_reasoning(prompt)` function rather than re-introducing the flag — the user-facing path is faster without it.
- **`` blocks.** The model may emit `` blocks inline. `_strip_thinking` handles this. If you change the prompt or model, watch for unstripped reasoning leaking into user-facing answers.
- **Deterministic point IDs are only deterministic if `content_hash` is present.** The chunker always sets it (sha256 of header + text), so this is safe in practice. Don't bypass the chunker and call `upsert_chunks` directly with chunks lacking `content_hash` unless you want random UUIDs.
- **`article_num` filter type.** `vector_store.search` filters on `article_num` as a **string** (it accepts the string form via Qdrant's `MatchValue`). The chunker stores it as `int`, but Qdrant will return it as `str` in the payload. Tests pass strings (`article_num="21"`) for consistency.

## What's next

Phase 3 — API. You'll wrap `ask()` in a FastAPI app with `/search`, `/ask`, `/health`, `/stats`, `/golden`, `/feedback` endpoints, all in a Docker container with Qdrant.
