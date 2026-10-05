import hashlib

from safety_rag.eval.corpus_pins import check_record_pins, count_term_hits, file_sha256


def test_file_sha256_matches_hashlib(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("hello world")
    assert file_sha256(p) == hashlib.sha256(b"hello world").hexdigest()


def test_count_term_hits_case_insensitive_by_default(tmp_path):
    p = tmp_path / "f.jsonl"
    p.write_text('{"text": "mentions GDPR here"}\n{"text": "no match"}\n')
    assert count_term_hits(p, "gdpr") == 1


def test_count_term_hits_exact_case_suffix(tmp_path):
    p = tmp_path / "f.jsonl"
    p.write_text('{"text": "Code of Practice exists"}\n{"text": "code of practice lowercase"}\n')
    assert count_term_hits(p, "Code of Practice (exact case)") == 1
    assert count_term_hits(p, "Code of Practice") == 2


def test_count_term_hits_counts_lines_not_occurrences(tmp_path):
    p = tmp_path / "f.jsonl"
    p.write_text('{"text": "DORA DORA DORA in one line"}\n{"text": "DORA again"}\n')
    assert count_term_hits(p, "DORA") == 2  # 2 matching lines, not 4 occurrences


def test_check_record_pins_detects_stale_hash(tmp_path):
    ai_act = tmp_path / "ai_act.jsonl"
    nis2 = tmp_path / "nis2.jsonl"
    ai_act.write_text('{"text": "x"}\n')
    nis2.write_text('{"text": "y"}\n')

    record = {
        "q_id": "u001",
        "corpus_snapshot": {"ai_act_sha256": "wrong", "nis2_sha256": "wrong"},
        "anchor_terms": {},
        "must_be_zero": [],
    }
    problems = check_record_pins(record, ai_act, nis2)
    assert any("corpus_snapshot" in p for p in problems)


def test_check_record_pins_clean_when_everything_matches(tmp_path):
    ai_act = tmp_path / "ai_act.jsonl"
    nis2 = tmp_path / "nis2.jsonl"
    ai_act.write_text('{"text": "no relevant terms"}\n')
    nis2.write_text('{"text": "also none"}\n')

    record = {
        "q_id": "u001",
        "corpus_snapshot": {"ai_act_sha256": file_sha256(ai_act), "nis2_sha256": file_sha256(nis2)},
        "anchor_terms": {"GDPR": [0, 0]},
        "must_be_zero": ["GDPR"],
    }
    assert check_record_pins(record, ai_act, nis2) == []


def test_check_record_pins_detects_must_be_zero_violation(tmp_path):
    ai_act = tmp_path / "ai_act.jsonl"
    nis2 = tmp_path / "nis2.jsonl"
    ai_act.write_text('{"text": "GDPR mentioned here now"}\n')
    nis2.write_text('{"text": "none"}\n')

    record = {
        "q_id": "u001",
        "corpus_snapshot": {"ai_act_sha256": file_sha256(ai_act), "nis2_sha256": file_sha256(nis2)},
        "anchor_terms": {},
        "must_be_zero": ["GDPR"],
    }
    problems = check_record_pins(record, ai_act, nis2)
    assert any("must_be_zero" in p for p in problems)
