from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import requests

CELLAR_URL = "https://publications.europa.eu/resource/celex/{}"
OUTPUT_DIR = Path("data/raw")


def sha256(path: Path) -> str:
    """Calculate the SHA-256 hash of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def download_document(celex: str) -> Path:
    """Download the English XHTML representation of a CELEX document."""
    url = CELLAR_URL.format(celex)
    output_path = OUTPUT_DIR / f"{celex}.html"

    print(f"Downloading {celex} from Cellar...")

    response = requests.get(
        url,
        headers={
            "Accept": "application/xhtml+xml",
            "Accept-Language": "eng",
            "Accept-Max-Cs-Size": str(200 * 1024 * 1024),
        },
        timeout=60,
    )

    print(f"HTTP status:  {response.status_code}")
    print(f"Content-Type: {response.headers.get('Content-Type')}")
    print(f"Size:         {len(response.content):,} bytes")

    response.raise_for_status()

    if not response.content:
        raise RuntimeError(
            f"Cellar returned an empty response for CELEX {celex}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download an EU legal document from EUR-Lex/Cellar."
    )
    parser.add_argument(
        "celex",
        help="CELEX number, e.g. 32024R1689",
    )

    args = parser.parse_args()

    path = download_document(args.celex)
    file_hash = sha256(path)

    print(f"Saved:        {path}")
    print(f"SHA-256:      {file_hash}")


if __name__ == "__main__":
    main()
