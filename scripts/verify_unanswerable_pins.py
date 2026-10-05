#!/usr/bin/env python
"""Re-verify the unanswerable set's corpus pins against the current corpus.

Run after any re-chunking or re-download — see the eval-drift pitfall
in phase-4-eval.md. A clean run confirms every anchor_terms count and
corpus_snapshot hash still matches data/processed/ as it stands today.

Usage:
    uv run python scripts/verify_unanswerable_pins.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from safety_rag.eval.corpus_pins import check_record_pins

DATA_DIR = Path("data/processed")
GOLDEN_PATH = Path("evals/golden/unanswerable.jsonl")


def main() -> int:
    ai_act_path = DATA_DIR / "ai_act.jsonl"
    nis2_path = DATA_DIR / "nis2.jsonl"
    records = [json.loads(l) for l in GOLDEN_PATH.read_text().splitlines() if l.strip()]

    problems = []
    for r in records:
        problems.extend(check_record_pins(r, ai_act_path, nis2_path))

    if problems:
        print(f"{len(problems)} drift issue(s) found:\n")
        for p in problems:
            print(f"  - {p}")
        return 1

    print(f"All {len(records)} records verified clean against the current corpus.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
