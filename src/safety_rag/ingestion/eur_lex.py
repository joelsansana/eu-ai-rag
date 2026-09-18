from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from selectolax.parser import HTMLParser

CELEX_TO_REGULATION = {
    "32024R1689": "ai_act",
    "32022L2555": "nis2",
}

CELEX_TO_EFFECTIVE_DATE = {
    "32024R1689": "2024-08-01",
    "32022L2555": "2023-01-16",
}


def _extract_number(value: str) -> int | None:
    match = re.search(r"\d+", value)
    return int(match.group()) if match else None


def _extract_title(node: Any) -> str:
    title_node = node.css_first(
        ".eli-title, .oj-ti-art, .oj-ti-section, .oj-ti-grseq"
    )

    if title_node is None:
        return ""

    return " ".join(title_node.text().split())


def _extract_paragraph_text(node: Any) -> str:
    paragraphs = node.css("p")

    return "\n\n".join(
        " ".join(paragraph.text().split())
        for paragraph in paragraphs
        if paragraph.text().strip()
    )


def _extract_chapter_text(node: Any) -> str:
    """Extract only the chapter's own heading and title.

    Chapters contain article subdivisions, so extracting all descendant
    paragraphs would duplicate the article text.
    """
    heading = node.css_first(".oj-ti-section-1")

    if heading is None:
        return ""

    title = node.css_first(".eli-title .oj-ti-section-2")

    parts = [" ".join(heading.text().split())]

    if title is not None:
        parts.append(" ".join(title.text().split()))

    return "\n\n".join(parts)


def _make_unit(
    *,
    regulation: str,
    part: str,
    celex: str,
    effective_date: str,
    title: str,
    text: str,
    html_path: Path,
    article_num: int | None = None,
    recital_num: int | None = None,
    annex_id: str | None = None,
    chapter: str | None = None,
) -> dict[str, Any]:
    return {
        "regulation": regulation,
        "part": part,
        "article_num": article_num,
        "recital_num": recital_num,
        "annex_id": annex_id,
        "chapter": chapter,
        "celex": celex,
        "effective_date": effective_date,
        "lang": "EN",
        "title": title,
        "text": text,
        "html_path": str(html_path),
    }


def parse_eur_lex(html_path: str | Path) -> list[dict[str, Any]]:
    path = Path(html_path)

    if not path.is_file():
        raise FileNotFoundError(f"EUR-Lex HTML file not found: {path}")

    celex = path.stem

    if celex not in CELEX_TO_REGULATION:
        raise ValueError(
            f"Unsupported CELEX '{celex}'. "
            f"Expected one of: {', '.join(CELEX_TO_REGULATION)}"
        )

    regulation = CELEX_TO_REGULATION[celex]
    effective_date = CELEX_TO_EFFECTIVE_DATE[celex]

    html = path.read_text(encoding="utf-8")
    document = HTMLParser(html)

    units: list[dict[str, Any]] = []

    # Recitals
    for node in document.css(".eli-subdivision[id^='rct_']"):
        node_id = node.attributes.get("id") or ""
        recital_num = _extract_number(node_id)

        text = _extract_paragraph_text(node)

        if not text:
            continue

        units.append(
            _make_unit(
                regulation=regulation,
                part="recital",
                recital_num=recital_num,
                celex=celex,
                effective_date=effective_date,
                title=_extract_title(node),
                text=text,
                html_path=path,
            )
        )

    # Articles
    for node in document.css(".eli-subdivision[id^='art_']"):
        node_id = node.attributes.get("id") or ""
        article_num = _extract_number(node_id)

        text = _extract_paragraph_text(node)

        if not text:
            continue

        units.append(
            _make_unit(
                regulation=regulation,
                part="article",
                article_num=article_num,
                celex=celex,
                effective_date=effective_date,
                title=_extract_title(node),
                text=text,
                html_path=path,
            )
        )

    # Chapters
    chapter_pattern = re.compile(r"^cpt_[IVXLCDM]+$")

    for node in document.css("div[id^='cpt_']"):
        node_id = node.attributes.get("id") or ""

        if not chapter_pattern.fullmatch(node_id):
            continue

        chapter_id = node_id.removeprefix("cpt_")

        text = _extract_chapter_text(node)

        if not text:
            continue

        units.append(
            _make_unit(
                regulation=regulation,
                part="chapter",
                chapter=chapter_id,
                celex=celex,
                effective_date=effective_date,
                title=text.split("\n\n", maxsplit=1)[0],
                text=text,
                html_path=path,
            )
        )

    # Annexes
    for node in document.css("div.eli-container[id^='anx_']"):
        node_id = node.attributes.get("id") or ""
        annex_id = node_id.removeprefix("anx_")

        text = _extract_paragraph_text(node)

        if not text:
            continue

        units.append(
            _make_unit(
                regulation=regulation,
                part="annex",
                annex_id=annex_id,
                celex=celex,
                effective_date=effective_date,
                title=_extract_title(node),
                text=text,
                html_path=path,
            )
        )

    return units
