# Phase 1 — Ingestion

> Corpus downloaded from EUR-Lex/Cellar, parsed, structure-aware chunked, and saved as versioned JSONL. ~5–6 hours.
>
> **Status (verified 2026-09-25):** this guide was corrected against the actual code after the build was completed. The material deltas: (a) the corpus is fetched from the **EU Publications Office Cellar endpoint** (`publications.europa.eu/resource/celex/{celex}`) with `Accept: application/xhtml+xml`, not the older `eur-lex.europa.eu/legal-content/EN/TXT/HTML/` URL; (b) the parser extracts **four** kinds of subdivisions (recitals, articles, chapters, annexes) — chapters and annexes are not mentioned in the original tutorial; (c) the chunker's `n_tokens` field uses a **regex-based estimator** (`_estimate_tokens`), not `tiktoken`, even though `tiktoken` remains a declared dependency for future use; (d) long Annexes are split by paragraph boundary when their estimated token count exceeds `LONG_ANNEX_TOKEN_THRESHOLD = 800`; (e) the pipeline saves **both** parsed records (optional, for debugging) and chunks to `data/processed/*.jsonl` — `scripts/build_index.py` reads only the chunks JSONL.

## Goal

Two JSONL files in `data/processed/` (filenames are arbitrary — `build_index.py` reads any `*.jsonl` in the directory):
- One file containing the **chunks** (the artefact the rest of the pipeline consumes)
- One file containing the **parsed records** (optional, useful for debugging the parser)

Each chunk is a JSON object with the schema documented in step 3 below.

Plus tests proving the chunker doesn't drop content, attaches correct metadata, and handles the known edge cases (Annex III row-chunking, Recitals, cross-references, empty subdivisions).

## Prerequisites

- Phase 0 done (CI green on `main`)
- Internet access (EU Cellar is reachable)
- No new dependencies beyond `requests` and `selectolax` (already in `pyproject.toml` from Phase 0)

```bash
uv sync --all-extras
```

`selectolax` is the HTML parser (faster than `lxml` for our use case; either works). `tiktoken` is declared in the dependency list but the current chunker uses a regex-based token estimator instead — see step 3 for why.

## Steps

### 1. Create the CELEX fetcher

Write `scripts/download_corpus.py`. It hits the **EU Publications Office Cellar** endpoint (NOT the older `eur-lex.europa.eu/legal-content/EN/TXT/HTML/` URL) and saves the XHTML to `data/raw/{celex}.html`.

```python
CELLAR_URL = "https://publications.europa.eu/resource/celex/{}"
OUTPUT_DIR = Path("data/raw")


def download_document(celex: str) -> Path:
    url = CELLAR_URL.format(celex)
    output_path = OUTPUT_DIR / f"{celex}.html"

    response = requests.get(
        url,
        headers={
            "Accept": "application/xhtml+xml",
            "Accept-Language": "eng",
            "Accept-Max-Cs-Size": str(200 * 1024 * 1024),
        },
        timeout=60,
    )
    response.raise_for_status()

    if not response.content:
        raise RuntimeError(
            f"Cellar returned an empty response for CELEX {celex}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)
    return output_path
```

The script is invoked as a CLI:

```bash
uv run python scripts/download_corpus.py 32024R1689
uv run python scripts/download_corpus.py 32022L2555
```

On success it prints the HTTP status, content-type, byte size, output path, and the **SHA-256** of the saved file. **Save the hashes** — you'll reference them in Phase 4 (the golden set is tied to a specific corpus version).

**Verify:** `data/raw/32024R1689.html` and `data/raw/32022L2555.html` exist; the printed SHA-256 hex digests are stable across re-runs (the Cellar endpoint returns the same bytes for a given CELEX).

### 2. Parse the AI Act into structured records

Write `src/safety_rag/ingestion/eur_lex.py`. The parser extracts **four** kinds of subdivisions from the EUR-Lex HTML, each with a different CSS selector:

| Subdivision | Selector | Notes |
|---|---|---|
| Recital | `.eli-subdivision[id^='rct_']` | Numbered by `_extract_number(node_id)` |
| Article | `.eli-subdivision[id^='art_']` | Numbered by `_extract_number(node_id)` |
| Chapter | `div[id^='cpt_']` filtered by regex `^cpt_[IVXLCDM]+$` | Roman numeral (`I`, `II`, …) |
| Annex | `div.eli-container[id^='anx_']` | **Different selector from articles** — uses `.eli-container` not `.eli-subdivision` |

Returns a list of dicts, each with this shape:

```python
{
    "regulation": "ai_act" | "nis2",
    "part": "recital" | "article" | "annex" | "chapter",
    "article_num": int | None,
    "recital_num": int | None,
    "annex_id": str | None,
    "chapter": str | None,
    "celex": str,
    "effective_date": str,  # YYYY-MM-DD; mapped via CELEX_TO_EFFECTIVE_DATE
    "lang": "EN",
    "title": str,
    "text": str,  # paragraphs joined by "\n\n"
    "html_path": str,  # pointer to the source file (str(Path))
}
```

CELEX → regulation / effective-date is hard-coded in the module:

```python
CELEX_TO_REGULATION = {
    "32024R1689": "ai_act",
    "32022L2555": "nis2",
}
CELEX_TO_EFFECTIVE_DATE = {
    "32024R1689": "2024-08-01",
    "32022L2555": "2023-01-16",
}
```

Adding a new regulation means adding to both maps.

**Defensive behaviour:**
- `parse_eur_lex` raises `FileNotFoundError` if the HTML path doesn't exist.
- `parse_eur_lex` raises `ValueError` if the CELEX (file stem) is not in the map.
- Records with empty `text` (e.g. an `<eli-subdivision>` with no `<p>` children) are skipped, not yielded as empty dicts.

Chapter text extraction is special: `_extract_chapter_text` returns **only the chapter heading and title** (`.oj-ti-section-1` + `.eli-title .oj-ti-section-2`), not the descendant article text. Otherwise the chapter record would duplicate everything inside it.

Title extraction for Annexes uses `.oj-doc-ti` because Annex headings and titles use a different element than Articles/Recitals — `.eli-title` / `.oj-ti-art` returns the annex number ("ANNEX I"), and the title is in the second `.oj-doc-ti` paragraph if present.

**Verify by smoke test:**

```python
from safety_rag.ingestion.eur_lex import parse_eur_lex

units = parse_eur_lex("data/raw/32024R1689.html")
print(f"{len(units)} units")
from collections import Counter

print(Counter(u["part"] for u in units))
```

The real AI Act HTML yields roughly 113 articles + 180 recitals + ~15 chapters + a handful of annexes (the exact counts depend on the consolidated version Cellar serves). For the synthetic fixture (`tests/fixtures/eur_lex/32024R1689.html`) the parser yields exactly 1 of each kind — see `tests/ingestion/test_eur_lex.py`.

### 3. Write the structure-aware chunker

Write `src/safety_rag/ingestion/chunker.py`. The chunker takes parsed records and emits retrieval chunks. **Articles and other normal structural units become one chunk each. Long Annexes are split into multiple chunks by paragraph boundary** when their estimated token count exceeds `LONG_ANNEX_TOKEN_THRESHOLD = 800`.

Chunk shape:

```python
{
    "chunk_id": str,  # sha256(regulation + article_num + position)[:12]
    "regulation": str,
    "part": str,
    "article_num": int | None,
    "recital_num": int | None,
    "annex_id": str | None,
    "chapter": str | None,
    "celex": str,
    "effective_date": str,
    "header": str,  # e.g. "Article 26, Regulation (EU) 2024/1689 (AI Act), effective 1 August 2024, Chapter 4 — Deployer obligations"
    "text": str,  # the chunk body, prefixed by the header
    "n_tokens": int,
    "content_hash": str,  # sha256 of (header + text) for reproducibility
}
```

**Header format** (built by `_make_header`):

| Part | Header format |
|---|---|
| Article | `Article {N}, {regulation_name}, effective {date}, Chapter {X} — {title}` (Chapter and title omitted if not present) |
| Recital | `Recital {N}, {regulation_name}, effective {date}` |
| Annex | `Annex {id}, {regulation_name}, effective {date} — {title}` |
| Chapter | `Chapter {id}, {regulation_name}, effective {date} — {title}` |
| Other | `{Part}, {regulation_name}, effective {date} — {title}` |

The chunker prepends the header to the text — this is a named technique from Anthropic's context-retrieval work and dramatically improves retrieval precision on structured corpora. **Don't skip this.**

Long-annex splits include a `— Part {position}` suffix in the header (position > 1 only) so each split is distinguishable.

**Token estimation** uses `_estimate_tokens` — a regex-based counter, **not `tiktoken`**:

```python
def _estimate_tokens(text: str) -> int:
    if not text.strip():
        return 0
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))
```

`len(text) // 4` is close enough if you want to avoid the dependency entirely; the chunker uses a regex for slightly better accuracy on punctuation-heavy regulatory text. `tiktoken` is kept in the dep list for the LLM-side use cases (counting prompt tokens before/after retrieval).

**Effective-date formatting:** `effective_date` in the record is `YYYY-MM-DD`; the chunker formats it as e.g. `1 August 2024` for the header.

**Verify by running it:**

```bash
uv run python -c "
from safety_rag.ingestion.chunker import chunk_records
import json
from pathlib import Path
records = [json.loads(line) for line in Path('data/processed/ai_act.parsed.jsonl').read_text().splitlines()]
chunks = chunk_records(records)
print(f'{len(chunks)} chunks')
print(chunks[0]['header'])
print(chunks[0]['text'][:300])
"
```

The exact chunk count depends on the corpus: ~300–400 for the AI Act (Articles + Recitals + Chapters + split Annexes), ~250–350 for NIS2.

### 4. Edge-case tests

Write `tests/ingestion/test_chunker.py` covering:

- **Annex III row-chunking** — build a record with `part="annex"`, `annex_id="III"`, text that's four repeated rows × 30. `chunk_records([record])` returns multiple chunks (≥2), every chunk has `part="annex"`, `annex_id="III"`, and every original row is preserved in the combined text.
- **Recital contextual header** — build a recital record. Chunk has `part="recital"`, `recital_num=42`, `header` starts with `Recital 42, Regulation (EU) 2024/1689 (AI Act), effective 1 August 2024`, `text` starts with `header`, original record text is present.
- **Cross-references preserved verbatim** — chunk text contains the cross-reference string exactly (no rewriting).
- **Empty subdivision skipped** — feeding an empty record alongside a valid one yields only the valid one (no crash).

Also write `tests/ingestion/test_eur_lex.py` against the synthetic fixture at `tests/fixtures/eur_lex/32024R1689.html`:

```python
AI_ACT_FIXTURE = Path("tests/fixtures/eur_lex/32024R1689.html")
```

Verify:
- The parser yields exactly 1 recital, 1 article, 1 chapter, 1 annex from the fixture.
- All records have `regulation="ai_act"`, `celex="32024R1689"`, `effective_date="2024-08-01"`, `lang="EN"`, `html_path=str(AI_ACT_FIXTURE)`.
- The recital has `recital_num=1` and `text` contains `"first recital"`.
- The article has `article_num=1` and `title == "Article 1"`.
- The chapter has `chapter="I"`, `title == "CHAPTER I"`, and `"GENERAL PROVISIONS"` in the text.
- The annex has `annex_id="I"`, `title == "List of Union harmonisation legislation"`, `"ANNEX I"` in the text.

**Run the suite:**

```bash
uv run pytest tests/ingestion/ -v
```

All tests must pass.

### 5. Write the pipeline (parse → chunk → save)

There's no dedicated `scripts/build_pipeline.py` — the parse/chunk/save step is short enough to live in a one-off Python invocation. The convention is:

1. Parse → save `data/processed/{regulation}.parsed.jsonl` (one line per record)
2. Chunk → save `data/processed/{regulation}.chunks.jsonl` (one line per chunk)

```bash
uv run python -c "
import json
from pathlib import Path
from safety_rag.ingestion.eur_lex import parse_eur_lex
from safety_rag.ingestion.chunker import chunk_records

for celex, regulation in [('32024R1689', 'ai_act'), ('32022L2555', 'nis2')]:
    html_path = Path(f'data/raw/{celex}.html')
    parsed = parse_eur_lex(html_path)
    chunks = chunk_records(parsed)

    Path('data/processed').mkdir(parents=True, exist_ok=True)
    Path(f'data/processed/{regulation}.parsed.jsonl').write_text(
        '\n'.join(json.dumps(r, ensure_ascii=False) for r in parsed) + '\n'
    )
    Path(f'data/processed/{regulation}.chunks.jsonl').write_text(
        '\n'.join(json.dumps(c, ensure_ascii=False) for c in chunks) + '\n'
    )
    print(f'{regulation}: {len(parsed)} records -> {len(chunks)} chunks')
"
```

**Verify:** `data/processed/{ai_act,nis2}.{parsed,chunks}.jsonl` exist; every line is valid JSON; `len(chunks)` matches the earlier Phase 1 step 3 count.

### 6. Commit

```bash
git add src/safety_rag/ingestion tests scripts/download_corpus.py
git commit -m "Phase 1: ingest EUR-Lex corpus (AI Act + NIS2)"
git push origin main
```

## Verify phase complete

- `data/raw/{32024R1689,32022L2555}.html` exist with content hashes logged
- `data/processed/{ai_act,nis2}.chunks.jsonl` exist with ~300–400 chunks each
- `uv run pytest -q` passes (the trivial Phase 0 test + the ingestion tests + the chunker edge-case tests)
- CI green on `main`

## Pitfalls

- **Use the Cellar endpoint, not the legacy EUR-Lex HTML URL.** The `eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{celex}` endpoint still works but serves HTML with a different structure; the parser is tuned for the Cellar XHTML output. If you switch URLs you'll need to retest all four selectors.
- **Annex selector is `.eli-container`, not `.eli-subdivision`.** This is the easiest mistake to make — articles and annexes both have `id` attributes but they're nested in different containers in the EUR-Lex DOM.
- **Chapter extraction must not include descendant paragraphs.** Chapters contain their articles, so `_extract_chapter_text` returns only the heading and title. Otherwise chapter records duplicate everything inside them.
- **Annex III is the trap.** It's a list with rows that look like Articles but aren't. Don't auto-number them as Articles. Tag them with `part=annex, annex_id=III` and let the long-annex split logic chunk by paragraph when the token estimate exceeds 800.
- **Header format depends on the regulation-name map.** Adding a new regulation requires updating `REGULATION_NAMES` in `chunker.py` (`{"ai_act": "Regulation (EU) 2024/1689 (AI Act)", "nis2": "Directive (EU) 2022/2555 (NIS2)"}`) so the header reads correctly.
- **`tiktoken` is in `pyproject.toml` but the chunker doesn't use it.** Token estimation is regex-based. `tiktoken` is reserved for the LLM-side token-budget accounting you'll need in Phase 2+ (e.g. truncating retrieved chunks to fit a prompt).
- **Don't write to `data/processed/` in tests.** Tests use fixture strings (synthetic records), not the corpus. The corpus is regeneratable but tests must be deterministic.
- **EUR-Lex HTML changes structure periodically.** Your CSS selectors are coupled to it. If a future re-fetch breaks parsing, you'll see empty chunk lists. Mitigation: log the chunk count and `assert` it stays in the expected range.

## What's next

Phase 2 — Retrieval v1. You'll wire the embedder, the Qdrant vector store, and the RAG `ask()` function that combines them.
