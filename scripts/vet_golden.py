#!/usr/bin/env python
"""Interactive review loop for golden-set candidates.

Usage:
    uv run python scripts/vet_golden.py --candidates evals/golden/ai_act.candidates.jsonl

Keys:
    l = accept into lepanto.jsonl
    d = accept into demo.jsonl
    e = edit the question text, then choose l/d
    r = reject
    s = skip for now (ask again next run)
    q = save and quit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safety_rag.eval.prefilter import prefilter
from safety_rag.eval.sampling import candidate_key, sample_unreviewed

GOLDEN_DIR = Path("evals/golden")
STATE_DIR = GOLDEN_DIR / "_vetting_state"


def load_chunk_lookup(regulation: str) -> dict[str, dict]:
    """content_hash -> chunk, so we can show full context during review."""
    path = Path("data/processed") / f"{regulation}.jsonl"
    lookup = {}
    for line in path.read_text().splitlines():
        if line.strip():
            chunk = json.loads(line)
            if chunk.get("content_hash"):
                lookup[chunk["content_hash"]] = chunk
    return lookup


def load_state(state_path: Path) -> dict[str, str]:
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {}


def save_state(state_path: Path, state: dict[str, str]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2))


def append_to_tier(tier: str, record: dict) -> None:
    out_path = GOLDEN_DIR / f"{tier}.jsonl"
    with out_path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def to_golden_record(candidate: dict, tier: str, q_id: str) -> dict:
    return {
        "q_id": q_id,
        "tier": tier,
        "question": candidate["question"],
        "gold_ids": [candidate["content_hash"]],
        "all_acceptable_ids": [candidate["content_hash"]],
        "difficulty": candidate.get("difficulty", "medium"),
    }


def next_q_id(tier: str) -> str:
    out_path = GOLDEN_DIR / f"{tier}.jsonl"
    n = 0
    if out_path.exists():
        n = sum(1 for line in out_path.read_text().splitlines() if line.strip())
    return f"q{n + 1:03d}"


def run(candidates_path: Path, regulation: str, sample_n: int | None = None, seed: int | None = None) -> None:
    candidates = [json.loads(l) for l in candidates_path.read_text().splitlines() if l.strip()]
    kept, dropped = prefilter(candidates)
    print(f"Loaded {len(candidates)} candidates, {dropped} auto-dropped by prefilter, {len(kept)} to review.\n")

    chunk_lookup = load_chunk_lookup(regulation)
    state_path = STATE_DIR / f"{candidates_path.stem}.state.json"
    state = load_state(state_path)

    if sample_n is not None:
        remaining = sample_unreviewed(kept, state, n=sample_n, seed=seed)
        print(f"Sampling {len(remaining)} unreviewed candidates (of {len(kept) - sum(1 for c in kept if state.get(candidate_key(c)))} unreviewed total).\n")
    else:
        remaining = [c for c in kept if state.get(candidate_key(c)) is None]
        print(f"{len(remaining)} not yet decided ({len(kept) - len(remaining)} already reviewed in a prior run).\n")

    for i, candidate in enumerate(remaining, start=1):
        chunk = chunk_lookup.get(candidate.get("content_hash"), {})
        print("=" * 70)
        print(f"[{i}/{len(remaining)}]  {chunk.get('header', '(header unknown)')}")
        print("-" * 70)
        text = chunk.get("text", "")
        print(text[:600] + ("..." if len(text) > 600 else ""))
        print("-" * 70)
        print(f"Q ({candidate.get('difficulty')}): {candidate['question']}")
        print("-" * 70)

        choice = input("[l]epanto  [d]emo  [e]dit  [r]eject  [s]kip  [q]uit > ").strip().lower()

        key = candidate_key(candidate)

        if choice == "q":
            save_state(state_path, state)
            print("Saved. Resume anytime — already-decided questions won't be shown again.")
            return

        if choice == "s":
            continue  # not recorded in state, will reappear next run

        if choice == "e":
            new_q = input("New question text: ").strip()
            if new_q:
                candidate["question"] = new_q
            choice = input("[l]epanto  [d]emo  [r]eject > ").strip().lower()

        if choice in ("l", "d"):
            tier = "lepanto" if choice == "l" else "demo"
            record = to_golden_record(candidate, tier, next_q_id(tier))
            append_to_tier(tier, record)
            state[key] = f"accepted:{tier}"
        elif choice == "r":
            state[key] = "rejected"
        else:
            print("Unrecognized key, treating as skip.")
            continue

        # save incrementally so a crash doesn't lose progress
        save_state(state_path, state)

    print("\nAll candidates reviewed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument(
        "--regulation",
        help="defaults to the candidates filename's prefix, e.g. ai_act.candidates.jsonl -> ai_act",
    )
    parser.add_argument("--sample", type=int, default=None, help="review a random batch of N unreviewed candidates instead of all of them")
    parser.add_argument("--seed", type=int, default=None, help="seed for reproducible sampling")
    args = parser.parse_args()

    regulation = args.regulation or args.candidates.stem.split(".")[0]
    run(args.candidates, regulation, sample_n=args.sample, seed=args.seed)


if __name__ == "__main__":
    main()
