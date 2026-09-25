from __future__ import annotations

from collections.abc import Sequence
from typing import Any

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


def _truncate(text: str, max_chars: int = MAX_CHUNK_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _format_source_header(payload: dict[str, Any]) -> str:
    """Build the '[Source N | ...]' header fields from a chunk's payload.
    Only includes fields that are present, since a chunk may be tied to an
    Article, a Recital, or an Annex rather than all three at once."""
    parts = [f"Regulation: {payload.get('regulation')}"]

    article_num = payload.get("article_num")
    if article_num:
        parts.append(f"Article: {article_num}")

    chapter = payload.get("chapter")
    if chapter:
        parts.append(f"Chapter: {chapter}")

    annex_id = payload.get("annex_id")
    if annex_id:
        parts.append(f"Annex: {annex_id}")

    recital_num = payload.get("recital_num")
    if recital_num:
        parts.append(f"Recital: {recital_num}")

    return " | ".join(parts)


def format_context(
    results: Sequence[Any], *, max_chunk_chars: int = MAX_CHUNK_CHARS
) -> str:
    """Build the {context} block for RAG_PROMPT from search results.

    `results` are expected to be Qdrant ScoredPoint objects (as returned by
    vector_store.search), each with a `.score` and a `.payload` dict
    containing at least 'text' and 'regulation'. Sorted by score, best
    first; each chunk's text is truncated to `max_chunk_chars`.
    """
    sorted_results = sorted(results, key=lambda r: r.score, reverse=True)

    blocks: list[str] = []
    for i, result in enumerate(sorted_results, start=1):
        payload = result.payload or {}
        header = _format_source_header(payload)
        text = _truncate(payload.get("text", ""), max_chunk_chars)
        blocks.append(f"[Source {i} | {header}]\n{text}")

    return "\n\n".join(blocks)


def build_prompt(
    question: str,
    results: Sequence[Any],
    *,
    max_chunk_chars: int = MAX_CHUNK_CHARS,
) -> str:
    """Assemble the full RAG_PROMPT with retrieved context and the question."""
    context = format_context(results, max_chunk_chars=max_chunk_chars)
    return RAG_PROMPT.format(context=context, question=question)
