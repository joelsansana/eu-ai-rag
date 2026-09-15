# Phase 0 — Setup

> Repo skeleton with CI running lint + tests on an empty `src/` tree. ~2–3 hours.

## Goal

A green GitHub Actions run on a commit that contains nothing but a `pyproject.toml`, an empty `src/safety_rag/__init__.py`, a passing trivial test, `.gitignore`, and a `.github/workflows/ci.yml`. If CI is green, your environment is correctly wired for everything that follows.

## Prerequisites

- Python 3.12 (`python --version`)
- `uv` installed (`uv --version`)
- `gh` authenticated (`gh auth status`)
- This repo cloned locally (`git clone … && cd eu-ai-rag`)

## Steps

### 1. Init `pyproject.toml` with `uv`

```bash
uv init --package --no-readme --python 3.12 --name safety-rag .
```

**Verify:** `pyproject.toml` exists, contains `[project]` with `name = "safety-rag"`, and `requires-python = ">=3.12"`.

### 2. Add dev tooling

```bash
uv add --dev ruff pyright pytest pytest-cov
```

**Verify:** `[dependency-groups]` section in `pyproject.toml` now lists `ruff`, `pyright`, `pytest`, `pytest-cov`.

### 3. Configure ruff and pyright

Add to `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM", "RUF"]

[tool.pyright]
pythonVersion = "3.12"
include = ["src", "tests"]
```

### 4. Create the package skeleton

```bash
mkdir -p src/safety_rag tests
touch src/safety_rag/__init__.py tests/__init__.py
echo 'def test_trivial(): assert True' > tests/test_smoke.py
```

**Verify:** `uv run pytest tests/ -v` shows `1 passed`.

### 5. Write `.gitignore`

```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.pyright_cache/
data/raw/
data/processed/
evals/results/
*.qdrant/
.env
.env.local
.DS_Store
```

**Verify:** `git status` shows only the files you intend to commit, not the ignored ones.

### 6. CI workflow

Create `.github/workflows/ci.yml`:

```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12"}
      - run: pip install uv
      - run: uv sync --all-extras
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run pyright src tests
      - run: uv run pytest -q
```

### 7. Push and watch

```bash
git add .
git commit -m "Phase 0: repo skeleton with CI"
git push origin main
```

Open the repo's Actions tab in the browser. **Verify:** green check within 2 minutes.

## Verify phase complete

- `uv run pytest` locally passes
- `uv run ruff check .` returns clean
- `uv run pyright src tests` returns 0 errors
- GitHub Actions shows green on `main`

## Commit

The single commit at the end of step 7 is the phase commit. No extra cleanup needed.

## Pitfalls

- **`uv init --package` vs `uv init`** — the `--package` flag creates the `src/safety_rag/` layout (what you want). Without it, you get a flat module.
- **Python version pinning** — keep `requires-python = ">=3.12"` and the GitHub Actions matrix pinned to 3.12. Drifting versions will bite you in Phase 4 (the eval harness depends on `numpy`/`pandas` ABI).
- **`.gitignore` for `data/`** — the corpus you'll download in Phase 1 is large and regeneratable. Committing it bloats the repo. The ignore is in step 5.
- **Secrets in CI** — don't put your `MINIMAX_API_KEY` in `ci.yml`. Use `secrets.MINIMAX_API_KEY` (Phase 4 will introduce this).

## What's next

Phase 1 — Ingestion. You'll download the EUR-Lex corpus, parse it, and write a structure-aware chunker.
