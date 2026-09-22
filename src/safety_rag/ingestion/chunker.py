from __future__ import annotations

import hashlib
import re
from typing import Any

# Annexes above this approximate token count are split into smaller chunks.
# This is deliberately conservative because Annexes can contain large tables.
LONG_ANNEX_TOKEN_THRESHOLD = 800


REGULATION_NAMES = {
    "ai_act": "Regulation (EU) 2024/1689 (AI Act)",
    "nis2": "Directive (EU) 2022/2555 (NIS2)",
}


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _estimate_tokens(text: str) -> int:
    """Estimate token count deterministically without a tokenizer dependency.

    This is intentionally simple. The value is useful for chunk sizing and
    metadata, but it is not intended to exactly reproduce a model tokenizer.
    """
    if not text.strip():
        return 0

    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def _format_effective_date(value: str) -> str:
    """Convert YYYY-MM-DD into a human-readable date."""
    year, month, day = value.split("-")

    month_names = (
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    )

    return f"{int(day)} {month_names[int(month) - 1]} {year}"


def _regulation_name(regulation: str) -> str:
    return REGULATION_NAMES.get(regulation, regulation)


def _make_header(unit: dict[str, Any], position: int | None = None) -> str:
    regulation = unit["regulation"]
    regulation_name = _regulation_name(regulation)
    effective_date = _format_effective_date(unit["effective_date"])

    part = unit["part"]
    chapter = unit.get("chapter")
    title = unit.get("title") or ""

    if part == "article":
        article_num = unit.get("article_num")

        header = (
            f"Article {article_num}, {regulation_name}, "
            f"effective {effective_date}"
        )

        if chapter:
            header += f", Chapter {chapter}"

        if title and title != f"Article {article_num}":
            header += f" — {title}"

    elif part == "recital":
        recital_num = unit.get("recital_num")

        header = (
            f"Recital {recital_num}, {regulation_name}, "
            f"effective {effective_date}"
        )

    elif part == "annex":
        annex_id = unit.get("annex_id")

        header = (
            f"Annex {annex_id}, {regulation_name}, effective {effective_date}"
        )

        if title:
            header += f" — {title}"

    elif part == "chapter":
        chapter_id = unit.get("chapter")

        header = (
            f"Chapter {chapter_id}, {regulation_name}, "
            f"effective {effective_date}"
        )

        if title:
            header += f" — {title}"

    else:
        header = (
            f"{part.title()}, {regulation_name}, effective {effective_date}"
        )

        if title:
            header += f" — {title}"

    if position is not None:
        header += f" — Part {position}"

    return header


def _make_chunk(
    unit: dict[str, Any],
    body: str,
    position: int,
) -> dict[str, Any]:
    header = _make_header(unit, position if position > 1 else None)
    text = f"{header}\n\n{body.strip()}"

    chunk_key = f"{unit['regulation']}{unit.get('article_num')}{position}"

    content_hash = _sha256(header + text)

    return {
        "chunk_id": _sha256(chunk_key)[:12],
        "regulation": unit["regulation"],
        "part": unit["part"],
        "article_num": unit.get("article_num"),
        "recital_num": unit.get("recital_num"),
        "annex_id": unit.get("annex_id"),
        "chapter": unit.get("chapter"),
        "celex": unit["celex"],
        "effective_date": unit["effective_date"],
        "header": header,
        "text": text,
        "n_tokens": _estimate_tokens(text),
        "content_hash": content_hash,
    }


def _split_annex_rows(text: str) -> list[str]:
    """Split an Annex into its existing paragraph/row-like units.

    The parser currently represents Annex content as paragraphs separated by
    blank lines. Keeping those boundaries gives us a useful approximation of
    table rows without introducing HTML-specific logic into the chunker.
    """
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def _chunk_long_annex(
    unit: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = _split_annex_rows(unit["text"])

    if not rows:
        return []

    chunks: list[dict[str, Any]] = []
    current_rows: list[str] = []
    position = 1

    for row in rows:
        candidate_rows = [*current_rows, row]
        candidate_body = "\n\n".join(candidate_rows)

        # Estimate using the eventual header as well. This prevents chunks
        # from exceeding the approximate target simply because of the header.
        candidate_header = _make_header(
            unit, position if position > 1 else None
        )
        candidate_text = f"{candidate_header}\n\n{candidate_body}"

        if (
            current_rows
            and _estimate_tokens(candidate_text) > LONG_ANNEX_TOKEN_THRESHOLD
        ):
            chunks.append(
                _make_chunk(
                    unit,
                    "\n\n".join(current_rows),
                    position,
                )
            )
            position += 1
            current_rows = [row]
        else:
            current_rows.append(row)

    if current_rows:
        chunks.append(
            _make_chunk(
                unit,
                "\n\n".join(current_rows),
                position,
            )
        )

    return chunks


def chunk_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert parsed EUR-Lex units into retrieval chunks.

    Articles and other normal structural units remain intact. Long Annexes
    are split on their existing paragraph/row boundaries.
    """
    chunks: list[dict[str, Any]] = []

    for unit in records:
        if not unit["text"].strip():
            continue

        if unit["part"] == "annex":
            token_count = _estimate_tokens(unit["text"])

            if token_count > LONG_ANNEX_TOKEN_THRESHOLD:
                chunks.extend(_chunk_long_annex(unit))
                continue

        chunks.append(
            _make_chunk(
                unit,
                unit["text"],
                1,
            )
        )

    return chunks
