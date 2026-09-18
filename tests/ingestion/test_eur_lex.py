from collections import Counter
from pathlib import Path

from safety_rag.ingestion.eur_lex import parse_eur_lex

AI_ACT_FIXTURE = Path("tests/fixtures/eur_lex/32024R1689.html")


def test_parse_ai_act_fixture() -> None:
    units = parse_eur_lex(AI_ACT_FIXTURE)

    counts = Counter(unit["part"] for unit in units)

    assert counts["recital"] == 1
    assert counts["article"] == 1
    assert counts["chapter"] == 1
    assert counts["annex"] == 1


def test_parse_ai_act_fixture_metadata() -> None:
    units = parse_eur_lex(AI_ACT_FIXTURE)

    for unit in units:
        assert unit["regulation"] == "ai_act"
        assert unit["celex"] == "32024R1689"
        assert unit["effective_date"] == "2024-08-01"
        assert unit["lang"] == "EN"
        assert unit["html_path"] == str(AI_ACT_FIXTURE)


def test_parse_ai_act_fixture_recital() -> None:
    units = parse_eur_lex(AI_ACT_FIXTURE)

    recital = next(unit for unit in units if unit["part"] == "recital")

    assert recital["recital_num"] == 1
    assert recital["title"] == ""
    assert "first recital" in recital["text"]


def test_parse_ai_act_fixture_article() -> None:
    units = parse_eur_lex(AI_ACT_FIXTURE)

    article = next(unit for unit in units if unit["part"] == "article")

    assert article["article_num"] == 1
    assert article["title"] == "Article 1"
    assert "first article" in article["text"]


def test_parse_ai_act_fixture_chapter() -> None:
    units = parse_eur_lex(AI_ACT_FIXTURE)

    chapter = next(unit for unit in units if unit["part"] == "chapter")

    assert chapter["chapter"] == "I"
    assert chapter["title"] == "CHAPTER I"
    assert "GENERAL PROVISIONS" in chapter["text"]


def test_parse_ai_act_fixture_annex() -> None:
    units = parse_eur_lex(AI_ACT_FIXTURE)

    annex = next(unit for unit in units if unit["part"] == "annex")

    assert annex["annex_id"] == "I"
    assert annex["title"] == "List of Union harmonisation legislation"
    assert "ANNEX I" in annex["text"]
    assert "Test annex content" in annex["text"]
