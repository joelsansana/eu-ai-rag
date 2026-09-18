from collections import Counter
from pathlib import Path

from safety_rag.ingestion.eur_lex import parse_eur_lex

AI_ACT_HTML = Path("data/raw/32024R1689.html")


def test_ai_act_parser_smoke():
    assert AI_ACT_HTML.exists(), f"Missing fixture: {AI_ACT_HTML}"

    units = parse_eur_lex(AI_ACT_HTML)
    counts = Counter(unit["part"] for unit in units)

    assert counts["recital"] == 180
    assert counts["article"] == 113
    assert counts["chapter"] == 13
    assert counts["annex"] == 13

    assert len(units) == 319

    assert all(unit["celex"] == "32024R1689" for unit in units)
    assert all(unit["regulation"] == "ai_act" for unit in units)
    assert all(unit["lang"] == "EN" for unit in units)
    assert all(unit["text"].strip() for unit in units)

    assert {unit["chapter"] for unit in units if unit["part"] == "chapter"} == {
        "I",
        "II",
        "III",
        "IV",
        "V",
        "VI",
        "VII",
        "VIII",
        "IX",
        "X",
        "XI",
        "XII",
        "XIII",
    }

    assert {unit["annex_id"] for unit in units if unit["part"] == "annex"} == {
        "I",
        "II",
        "III",
        "IV",
        "V",
        "VI",
        "VII",
        "VIII",
        "IX",
        "X",
        "XI",
        "XII",
        "XIII",
    }
