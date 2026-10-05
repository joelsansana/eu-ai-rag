import pytest
import json
from pathlib import Path

from safety_rag.eval.schemas import UNANSWERABLE_CATEGORIES, UnanswerableQuestion

GOLDEN_DIR = Path("evals/golden")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


@pytest.fixture
def unanswerable_records():
    return _load_jsonl(GOLDEN_DIR / "unanswerable.jsonl")


def test_each_record_validates(unanswerable_records):
    for r in unanswerable_records:
        UnanswerableQuestion.model_validate(r)


def test_q_ids_are_unique(unanswerable_records):
    ids = [r["q_id"] for r in unanswerable_records]
    assert len(ids) == len(set(ids))


def test_questions_are_unique(unanswerable_records):
    qs = [r["question"].strip().lower() for r in unanswerable_records]
    assert len(qs) == len(set(qs))


def test_between_eight_and_twelve_questions(unanswerable_records):
    assert 8 <= len(unanswerable_records) <= 12


def test_categories_are_known(unanswerable_records):
    for r in unanswerable_records:
        assert r["category"] in UNANSWERABLE_CATEGORIES


def test_no_overlap_with_golden_questions(unanswerable_records):
    golden_qs = set()
    for tier_file in ("lepanto.jsonl", "demo.jsonl"):
        for r in _load_jsonl(GOLDEN_DIR / tier_file):
            golden_qs.add(r["question"].strip().lower())

    unanswerable_qs = {r["question"].strip().lower() for r in unanswerable_records}
    assert not (golden_qs & unanswerable_qs)
