# Phase 3 — API

> FastAPI app in Docker, with `/ask` streaming via SSE. ~5–6 hours.

## Goal

`docker compose up` starts the app and Qdrant. `curl http://localhost:8000/ask -d '{"q":"…"}'` returns a streamed answer with citations.

## Prerequisites

- Phase 2 done
- Docker + Docker Compose
- `data/processed/*.jsonl` populated, Qdrant collection `safety_rag_v1` built (re-run `scripts/build_index.py` if needed)
- `MINIMAX_API_KEY` available at runtime (via `.env` or compose secrets)

## Steps

### 1. Add FastAPI and SSE

```bash
uv add fastapi uvicorn sse-starlette pydantic
```

### 2. Pydantic schemas

Create `src/safety_rag/api/schemas.py`:

```python
from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=20)
    regulation: str | None = None       # "ai_act" | "nis2" | None
    article_num: int | None = None
    part: str | None = None             # "recital" | "article" | "annex"

class SourceChunk(BaseModel):
    score: float
    regulation: str
    part: str
    article_num: int | None
    recital_num: int | None
    annex_id: str | None
    chapter: str | None
    celex: str
    header: str
    text: str
    content_hash: str

class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    latency_ms: int

class SearchRequest(BaseModel):
    question: str
    k: int = 5
    regulation: str | None = None
    article_num: int | None = None
    part: str | None = None
```

### 3. Auth (env-var API token)

```python
# src/safety_rag/api/auth.py
import os
from fastapi import Header, HTTPException

async def require_token(authorization: str | None = Header(default=None)):
    expected = os.environ.get("API_TOKEN")
    if not expected:
        return  # dev mode: no token required
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    if authorization.removeprefix("Bearer ").strip() != expected:
        raise HTTPException(status_code=401, detail="Invalid token")
```

Add `API_TOKEN=*** to your `.env`. Do not commit `.env`.

### 4. Endpoints

Create `src/safety_rag/api/main.py`:

```python
from fastapi import FastAPI, Depends
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
import time, json

from safety_rag.api.schemas import AskRequest, AskResponse, SearchRequest, SourceChunk
from safety_rag.api.auth import require_token
from safety_rag.api.ask import ask
from safety_rag.retrieval.vector_store import search, embed

app = FastAPI(title="eu-ai-rag", version="0.1.0")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/search", dependencies=[Depends(require_token)])
async def search_endpoint(req: SearchRequest):
    q_emb = embed([req.question])[0]
    results = search(q_emb, k=req.k, regulation=req.regulation,
                     article_num=req.article_num, part=req.part)
    return {"results": [
        SourceChunk(score=r["score"], **r["payload"]) for r in results
    ]}

@app.post("/ask", dependencies=[Depends(require_token)])
async def ask_endpoint(req: AskRequest):
    async def event_stream():
        t0 = time.monotonic()
        result = ask(req.question, k=req.k, regulation=req.regulation,
                     article_num=req.article_num, part=req.part)
        yield {"event": "answer", "data": result["answer"]}
        yield {"event": "sources", "data": json.dumps(
            [SourceChunk(score=r["score"], **r["payload"]).model_dump()
             for r in result["sources"]],
            default=str
        )}
        yield {"event": "done", "data": json.dumps(
            {"latency_ms": int((time.monotonic() - t0) * 1000)}
        )}
    return EventSourceResponse(event_stream())
```

SSE event flow:
1. `answer` — the generated text
2. `sources` — JSON array of cited chunks
3. `done` — latency

### 5. Multi-stage Dockerfile

Create `Dockerfile`:

```dockerfile
FROM python:3.12-slim AS builder
WORKDIR /app
RUN pip install uv
COPY pyproject.toml uv.lock ./
RUN uv export --no-dev --frozen -o requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim AS runtime
WORKDIR /app
RUN useradd --create-home appuser
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY src ./src
COPY data/processed ./data/processed
USER appuser
EXPOSE 8000
CMD ["uvicorn", "safety_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build:
```bash
docker build -t eu-ai-rag:latest .
```

### 6. docker-compose

Create `docker-compose.yml`:

```yaml
services:
  qdrant:
    image: qdrant/qdrant
    ports: ["6333:6333"]
    volumes: ["qdrant_data:/qdrant/storage"]
  app:
    build: .
    depends_on: [qdrant]
    ports: ["8000:8000"]
    env_file: .env
    command: >
      sh -c "python scripts/build_index.py &&
             uvicorn safety_rag.api.main:app --host 0.0.0.0 --port 8000"
volumes:
  qdrant_data:
```

### 7. Smoke test

```bash
docker compose up -d
sleep 30  # let Qdrant + app boot
curl http://localhost:8000/health
# {"status":"ok"}

curl -N -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ***" \
  -d '{"question":"What are the deployer obligations under Article 26 of the AI Act?"}'
```

Expected: SSE stream of `answer`, `sources`, `done` events.

### 8. Integration tests with Qdrant in-memory

In `tests/test_api.py`:

```python
import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    from safety_rag.api.main import app
    return TestClient(app)

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
```

For the `/ask` test, mock the MiniMax client (don't burn real API credits in CI). Use `monkeypatch` to replace `safety_rag.generation.llm.generate` with a fixed response.

### 9. Commit

```bash
git add src/safety_rag/api scripts tests Dockerfile docker-compose.yml
git commit -m "Phase 3: FastAPI app in Docker"
git push origin main
```

## Verify phase complete

- `docker compose up` brings up Qdrant + app
- `/health` returns 200
- `/ask` returns SSE stream with answer + sources
- `curl` without Bearer token returns 401
- CI green

## Pitfalls

- **`requirements.txt` lock drift.** Re-run `uv lock` after every dep change. The Dockerfile uses `--frozen` which will fail loudly if the lock is stale — good.
- **SSE in Docker.** SSE needs `--buffer-size 0` or streaming responses. uvicorn handles this by default for `EventSourceResponse`. If you see buffering, set `--no-buffer`.
- **Embedding model download in Docker.** First build downloads ~100MB of model weights. Cache them with a Docker build stage or pre-pull into the image. The model is cached at `~/.cache/huggingface/`; mount that as a volume in dev.
- **Qdrant data persistence.** Without the named volume in `docker-compose.yml`, Qdrant loses data on container restart. Don't skip the volume.
- **`API_TOKEN` length.** 32 random bytes (base64) is fine. Don't make it short — `Authorization: Bearer` *** the only auth, no rate limiting in Phase 3.

## What's next

Phase 4 — Evaluation. You'll generate a golden set of ~50 questions, write retrieval + generation metrics, and add a CI regression gate that fails if Hit@5 drops >2 points.
