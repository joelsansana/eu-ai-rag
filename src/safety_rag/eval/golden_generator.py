from __future__ import annotations

import json

from safety_rag.generation.llm import generate

GOLDEN_PROMPT = """You are creating evaluation questions for a regulatory RAG system.

Given the following legal text chunk, generate exactly 3 distinct natural-language questions that this chunk, and only this chunk, would answer. Vary the phrasing and specificity so the questions are not near-duplicates of each other.

Chunk header: {header}
Chunk text:
{text}

Respond with ONLY a JSON array, no other text, no markdown fences, in this exact shape:
[
  {{"question": "...", "difficulty": "easy" | "medium" | "hard", "answerable": true}},
  {{"question": "...", "difficulty": "easy" | "medium" | "hard", "answerable": true}},
  {{"question": "...", "difficulty": "easy" | "medium" | "hard", "answerable": true}}
]
"""

_RETRY_SUFFIX = (
    "\n\nReminder: output ONLY the JSON array. "
    "No prose, no markdown code fences, no explanation."
)


class GoldenGenerationError(Exception):
    """Raised when the model fails to produce parseable JSON after a retry."""


def generate_candidates(chunk: dict, *, temperature: float = 0.2) -> list[dict]:
    """Generate ~3 candidate golden questions for a single chunk.

    Each candidate is annotated with the chunk's content_hash, regulation,
    and celex so manual vetting can trace it back to a specific corpus
    chunk (see the eval-drift pitfall — this linkage is load-bearing).
    """
    prompt = GOLDEN_PROMPT.format(header=chunk.get("header") or "", text=chunk["text"])

    raw = generate(prompt, temperature=temperature)
    candidates = _parse_json_array(raw)

    if candidates is None:
        raw_retry = generate(prompt + _RETRY_SUFFIX, temperature=temperature)
        candidates = _parse_json_array(raw_retry)

    if candidates is None:
        raise GoldenGenerationError(
            f"could not parse JSON for chunk {chunk.get('content_hash')!r}"
        )

    for c in candidates:
        c["content_hash"] = chunk.get("content_hash")
        c["regulation"] = chunk.get("regulation")
        c["celex"] = chunk.get("celex")

    return candidates


def _parse_json_array(text: str) -> list[dict] | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    return data if isinstance(data, list) else None
