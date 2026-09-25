from __future__ import annotations

from typing import Any

from safety_rag.generation.llm import generate
from safety_rag.generation.prompts import build_prompt
from safety_rag.retrieval.embedder import embed
from safety_rag.retrieval.vector_store import search


def ask(question: str, *, k: int = 5, **filters: Any) -> dict[str, Any]:
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
