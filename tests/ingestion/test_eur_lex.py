from collections import Counter
from pathlib import Path

from safety_rag.ingestion.eur_lex import parse_eur_lex

AI_ACT_HTML = Path("data/raw/32024R1689.html")


def test_parse_ai_act_counts() -> None:
    units = parse_eur_lex(AI_ACT_HTML)

    counts = Counter(unit["part"] for unit in units)

    assert counts["recital"] == 180
    assert counts["article"] == 113
    assert counts["chapter"] == 13
    assert counts["annex"] == 13
    assert len(units) == 319


def test_parse_ai_act_metadata() -> None:
    units = parse_eur_lex(AI_ACT_HTML)

    assert units

    for unit in units:
        assert unit["regulation"] == "ai_act"
        assert unit["celex"] == "32024R1689"
        assert unit["effective_date"] == "2024-08-01"
        assert unit["lang"] == "EN"
        assert unit["html_path"] == str(AI_ACT_HTML)


def test_parse_ai_act_recitals() -> None:
    units = parse_eur_lex(AI_ACT_HTML)

    recitals = [unit for unit in units if unit["part"] == "recital"]

    assert recitals[0]["recital_num"] == 1
    assert recitals[0]["title"] == ""
    assert recitals[0]["text"]


def test_parse_ai_act_articles() -> None:
    units = parse_eur_lex(AI_ACT_HTML)

    articles = [unit for unit in units if unit["part"] == "article"]

    assert articles[0]["article_num"] == 1
    assert articles[0]["title"]
    assert articles[0]["text"]


def test_parse_ai_act_chapters() -> None:
    units = parse_eur_lex(AI_ACT_HTML)

    chapters = [unit for unit in units if unit["part"] == "chapter"]

    assert chapters[0]["chapter"] == "I"
    assert chapters[0]["title"] == "CHAPTER I"
    assert "GENERAL PROVISIONS" in chapters[0]["text"]


def test_parse_ai_act_annexes() -> None:
    units = parse_eur_lex(AI_ACT_HTML)

    annexes = [unit for unit in units if unit["part"] == "annex"]

    assert annexes[0]["annex_id"] == "I"
    assert annexes[0]["title"]
    assert "ANNEX I" in annexes[0]["text"]
