# Phase 1 — Ingestion

> Corpus downloaded from EUR-Lex, parsed, structure-aware chunked, and saved as versioned JSONL. ~5–6 hours.

## Goal

Two JSONL files in `data/processed/`:
- `ai_act.jsonl` — every Article, Recital, and Annex of the AI Act, one chunk per line, with metadata
- `nis2.jsonl` — same for NIS2

Plus tests proving the chunker doesn't drop content, attaches correct metadata, and handles the known edge cases (Annex III, Recitals, cross-references).

## Prerequisites

- Phase 0 done (CI green on `main`)
- Internet access (EUR-Lex is reachable)
- No new dependencies beyond `requests` and `selectolax` (add via `uv add`)

```bash
uv add requests selectolax tiktoken
```

`tiktoken` is for token counting during chunking. `selectolax` is the HTML parser (faster than `lxml` for our use case; either works).

## Steps

### 1. Create the CELEX fetcher

Write `scripts/download_corpus.py`. It should:

- Take a CELEX number (`32024R1689` for AI Act, `32022L2555` for NIS2)
- Hit EUR-Lex's HTML endpoint: `https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{celex}`
- Save raw HTML to `data/raw/{celex}.html`
- Compute and print a SHA-256 hash of the file (so you can detect drift later)

**Run it:**
```bash
uv run python scripts/download_corpus.py 32024R1689
uv run python scripts/download_corpus.py 32022L2555
```

**Verify:** `data/raw/32024R1689.html` and `data/raw/32022L2555.html` exist, both >100KB. Print the hashes and save them — you'll reference them in Phase 4 (the golden set is tied to a specific corpus version).

### 2. Parse the AI Act into structured records

Write `src/safety_rag/ingestion/eur_lex.py`. It should:

- Take a path to the saved HTML
- Return a list of dicts, one per Article/Recital/Annex/Chapter, each with:

```python
{
    "regulation": "ai_act" | "nis2",
    "part": "recital" | "article" | "annex" | "chapter",
    "article_num": int | None,
    "recital_num": int | None,
    "annex_id": str | None,
    "chapter": str | None,
    "celex": str,
    "effective_date": str,
    "lang": "EN",
    "title": str,
    "text": str,  # the raw text content of the unit
    "html_path": str,  # pointer to the source file
}
```

EUR-Lex HTML structure (consistent across regulations):
- Recitals are inside `<div class="eli-subdivision" id="rct_…">`
- Articles are inside `<div class="eli-subdivision" id="art_…">`
- Annexes are inside `<div class="eli-subdivision" id="anx_…">`
- Chapters are inside `<div class="eli-subdivision" id="cpt_…">`

Use `selectolax` CSS selectors (`.eli-subdivision[id^='art_']`, etc.) to extract them. For the text content, concatenate the `<p>` elements within each subdivision.

**Verify by smoke test:** running the parser on `32024R1689.html` should yield ~150–200 records (the AI Act has 113 Articles + 180 Recitals + Annexes).

### 3. Write the structure-aware chunker

Write `src/safety_rag/ingestion/chunker.py`. For most cases, one Article = one chunk. For very long Annexes (Annex III of the AI Act is a table-heavy list of high-risk AI-system categories), chunk by row.

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
    "header": str,  # "Article 26, Regulation (EU) 2024/1689 (AI Act), effective 2 August 2026, Chapter 4 — Deployer obligations"
    "text": str,  # the chunk body, prefixed by the header
    "n_tokens": int,
    "content_hash": str,  # sha256 of (header + text) for reproducibility
}
```

The **header** is prepended to the text. This is a named technique from Anthropic's context-retrieval work — contextual chunk headers dramatically improve retrieval precision on structured corpora. Don't skip this.

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

Should print a number in the 200–400 range, with a header like `Article 26, Regulation (EU) 2024/1689 (AI Act), …`.

### 4. Edge-case tests

Write `tests/test_chunker.py` covering:
- **Annex III row-chunking** — the high-risk AI-system categories list
- **Recitals** — short paragraphs, no Article heading; header should still be set (`Recital 42, Regulation (EU) 2024/1689 (AI Act), …`)
- **Cross-references** — Articles that explicitly cite another Article; the chunk text should include the cross-reference verbatim (no rewriting)
- **Empty subdivisions** — defensive: if EUR-Lex has a placeholder subdivision with no `<p>` elements, don't crash

```bash
uv run pytest tests/test_chunker.py -v
```

All four tests must pass.

### 5. Write the full pipeline

Wire `scripts/build_index.py`:
1. Load `data/raw/*.html`
2. Parse → records (step 2)
3. Chunk → chunks (step 3)
4. Save `data/processed/ai_act.jsonl` and `data/processed/nis2.jsonl`

```bash
uv run python scripts/build_index.py
```

**Verify:** `data/processed/ai_act.jsonl` and `data/processed/nis2.jsonl` exist, line counts printed, each line valid JSON.

### 6. Commit

```bash
git add src/safety_rag/ingestion scripts tests
git commit -m "Phase 1: ingest EUR-Lex corpus (AI Act + NIS2)"
git push origin main
```

## Verify phase complete

- `data/raw/{32024R1689,32022L2555}.html` exist with content hashes logged
- `data/processed/{ai_act,nis2}.jsonl` exist with 200–400 chunks each
- `uv run pytest -q` passes (trivially in Phase 0, plus your 4 chunker tests)
- CI green on `main`

## Pitfalls

- **EUR-Lex HTML changes structure periodically.** Your CSS selectors are coupled to it. If a future re-fetch breaks parsing, you'll see empty chunk lists. Mitigation: log the chunk count and `assert` it stays in the expected range.
- **Annex III is the trap.** It's a list with rows that look like Articles but aren't. Don't auto-number them as Articles. Tag them with `part=annex, annex_id=III` and chunk by row.
- **Headers are not optional.** Without the contextual chunk header, dense retrieval will treat every Article as a generic regulation paragraph. The header is what makes Article 26 distinguishable from Article 27 in embedding space.
- **Token count is approximate.** `tiktoken` is fast but you don't need byte-level accuracy. `len(text) // 4` is close enough if you want to avoid the dependency.
- **Don't write to `data/processed/` in tests.** Tests should use fixture strings, not hit the corpus. The corpus is regeneratable but the tests should be deterministic.

## What's next

Phase 2 — Retrieval v1. You'll embed the chunks with `bge-small-en-v1.5`, upsert to Qdrant, and write a naive RAG function with MiniMax-M2 as the generator.
