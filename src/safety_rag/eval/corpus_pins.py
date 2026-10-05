from __future__ import annotations

import hashlib
import re
from pathlib import Path

EXACT_SUFFIX = " (exact case)"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def count_term_hits(path: Path, term: str) -> int:
    """Count JSONL lines (chunks) containing term — the project's
    documented verification method: grep -ci, case-sensitive if the term
    is suffixed ' (exact case)'."""
    exact = term.endswith(EXACT_SUFFIX)
    bare_term = term[: -len(EXACT_SUFFIX)] if exact else term
    flags = 0 if exact else re.IGNORECASE
    pattern = re.compile(re.escape(bare_term), flags)

    count = 0
    for line in path.read_text().splitlines():
        if line.strip() and pattern.search(line):
            count += 1
    return count


def check_record_pins(record: dict, ai_act_path: Path, nis2_path: Path) -> list[str]:
    """Returns human-readable problems; empty list means the record's
    pins are consistent with the corpus as it stands right now."""
    problems: list[str] = []
    q_id = record.get("q_id", "?")

    current = {
        "ai_act_sha256": file_sha256(ai_act_path),
        "nis2_sha256": file_sha256(nis2_path),
    }
    snap = record.get("corpus_snapshot", {})
    for key in ("ai_act_sha256", "nis2_sha256"):
        if snap.get(key) != current[key]:
            problems.append(f"{q_id}: corpus_snapshot[{key}] is stale")

    for term, recorded in record.get("anchor_terms", {}).items():
        actual = [count_term_hits(ai_act_path, term), count_term_hits(nis2_path, term)]
        if list(recorded) != actual:
            problems.append(f"{q_id}: anchor_terms[{term!r}] recorded {recorded}, actual {actual}")

    for term in record.get("must_be_zero", []):
        actual = [count_term_hits(ai_act_path, term), count_term_hits(nis2_path, term)]
        if actual != [0, 0]:
            problems.append(f"{q_id}: must_be_zero term {term!r} now has hits {actual} — question may have become answerable")

    return problems
