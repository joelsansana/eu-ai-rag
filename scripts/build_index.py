from __future__ import annotations

import json
from pathlib import Path

from safety_rag.ingestion.chunker import chunk_records
from safety_rag.ingestion.eur_lex import parse_eur_lex

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CORPUS = {
    "ai_act": "32024R1689.html",
    "nis2": "32022L2555.html",
}


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_index(regulation: str, filename: str) -> None:
    raw_path = RAW_DIR / filename
    output_path = PROCESSED_DIR / f"{regulation}.jsonl"

    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw corpus file not found: {raw_path}")

    parsed_records = parse_eur_lex(raw_path)
    chunks = chunk_records(parsed_records)

    write_jsonl(output_path, chunks)

    print(f"{regulation}:")
    print(f"  Parsed records: {len(parsed_records)}")
    print(f"  Chunks:         {len(chunks)}")
    print(f"  Output:         {output_path}")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for regulation, filename in CORPUS.items():
        build_index(regulation, filename)


if __name__ == "__main__":
    main()
