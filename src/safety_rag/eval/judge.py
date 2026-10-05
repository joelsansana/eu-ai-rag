from __future__ import annotations

import json
import logging

from safety_rag.generation.llm import generate

logger = logging.getLogger(__name__)

_ANSWERABLE_CRITERIA = """\
- faithfulness: "yes" | "no" | "partial" — Is every claim in the answer
  supported by the retrieved context? If "no" or "partial", quote each
  unsupported span in "evidence".
- citation_accuracy: "yes" | "no" — Does each cited Article/recital in the
  answer actually contain the claim it is cited for? Compare the cited
  source text against the claim."""

_ABSTENTION_CRITERIA = """\
- abstention: "yes" | "no" — The question is unanswerable from the retrieved
  context. Did the answer correctly decline to answer (e.g. "the corpus does
  not address this") rather than fabricate one?"""

_VERDICT_SHAPE_ANSWERABLE = """\
{"faithfulness": "yes" | "no" | "partial", "citation_accuracy": "yes" | "no", "evidence": "..."}"""

_VERDICT_SHAPE_ABSTENTION = """\
{"abstention": "yes" | "no", "evidence": "..."}"""

_ALLOWED = {
    "faithfulness": {"yes", "no", "partial"},
    "citation_accuracy": {"yes", "no"},
    "abstention": {"yes", "no"},
}

_RETRY_SUFFIX = (
    "\n\nReminder: output ONLY the JSON object. "
    "No prose, no markdown code fences, no explanation."
)


class JudgeError(Exception):
    """Raised when the judge fails to produce a valid verdict after a retry."""


def judge(
    question: str,
    sources: list[dict],
    answer: str,
    *,
    expect_abstention: bool = False,
    temperature: float = 0.2,
) -> dict:
    """Judge one RAG answer against its retrieved context.

    expect_abstention=False  -> verdict: faithfulness, citation_accuracy
    expect_abstention=True   -> verdict: abstention

    Raises JudgeError after one stricter retry if the output is unparseable
    or uses a value outside the allowed set. The runner should catch this
    and mark the question unjudged (same failure-tolerant pattern as
    run_eval.py's n_failed).
    """
    criteria = _ABSTENTION_CRITERIA if expect_abstention else _ANSWERABLE_CRITERIA
    shape = _VERDICT_SHAPE_ABSTENTION if expect_abstention else _VERDICT_SHAPE_ANSWERABLE

    prompt = f"""You are an evaluator for a regulatory RAG system over the EU regulation.

Question: {question}

Retrieved context (numbered sources with their headers):
{_format_sources(sources)}

Answer: {answer}

For each criterion below, respond with JSON only:
{criteria}

The "evidence" field is a short quote or justification for your verdicts.

Respond with ONLY the JSON object, no other text, no markdown fences, in this exact shape:
{shape}"""

    verdict = _parse_and_validate(generate(prompt, temperature=temperature), expect_abstention)
    if verdict is None:
        verdict = _parse_and_validate(
            generate(prompt + _RETRY_SUFFIX, temperature=temperature), expect_abstention
        )
    if verdict is None:
        raise JudgeError(f"judge produced no valid verdict for question: {question[:80]!r}")
    return verdict


def _format_sources(sources: list[dict]) -> str:
    parts = []
    for i, s in enumerate(sources, 1):
        header = s.get("header") or "(no header)"
        text = s.get("text") or ""
        parts.append(f"[{i}] {header}\n{text}")
    return "\n\n".join(parts)


def _parse_and_validate(raw: str, expect_abstention: bool) -> dict | None:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    required = ["abstention"] if expect_abstention else ["faithfulness", "citation_accuracy"]
    for key in required:
        if data.get(key) not in _ALLOWED[key]:
            return None
    if not isinstance(data.get("evidence", ""), str):
        return None
    return data
