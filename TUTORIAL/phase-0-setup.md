# Phase 0 — Setup

> Repo skeleton with CI running lint + tests on an empty `src/` tree. ~2–3 hours.
>
> **Status (verified 2026-09-25):** this guide was corrected against the actual code after the build was completed. The two material deltas: (a) the project uses `uv_build` as the build backend with `uv sync --all-extras` as the install command (not `uv init --package` + manual `uv add`); (b) `ruff` is configured with `line-length = 80` (the tutorial originally said 100). Everything else is the same shape.

## Goal

A green GitHub Actions run on a commit that contains `pyproject.toml`, `src/safety_rag/__init__.py` (with a `main()` entry point), a passing trivial test, `.gitignore`, and `.github/workflows/ci.yml`. If CI is green, your environment is correctly wired for everything that follows.

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

**Verify:** `pyproject.toml` exists, contains `[project]` with `name = "safety-rag"`, and `requires-python = ">=3.12"`. The build backend is `uv_build` (`build-backend = "uv_build"`) and the project script `safety-rag = "safety_rag:main"` is declared in `[project.scripts]`.

### 2. Add dependencies (prod + dev)

The dependency list lives in `pyproject.toml`. After the `uv init` step, edit the file to add the runtime and dev dependencies declared by the build:

```toml
dependencies = [
    "httpx>=0.28.1",
    "openai>=3.17.0",
    "qdrant-client>=1.19.1",
    "requests>=2.34.2",
    "selectolax>=0.4.11",
    "sentence-transformers>=6.1.0",
    "tiktoken>=0.14.0",
]

[dependency-groups]
dev = [
    "pyright>=1.1.414",
    "pytest>=9.1.1",
    "pytest-cov>=7.1.0",
    "ruff>=0.16.8",
]
```

Then install everything (including the `dev` group):

```bash
uv sync --all-extras
```

`uv sync --all-extras` resolves the lockfile (`uv.lock`) and installs both runtime and dev dependencies in the project virtualenv. The tutorial's older `uv add --dev ruff pyright pytest pytest-cov` pattern still works but is now superseded by the declarative `pyproject.toml` + `uv sync` flow.

### 3. Configure ruff and pyright

The actual ruff config uses **`line-length = 80`** (not 100 as a typical project default). Pyright targets Python 3.12, includes `src` and `tests`, and is strict by default. Add to `pyproject.toml`:

```toml
[tool.ruff]
line-length = 80
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM", "RUF"]

[tool.pyright]
pythonVersion = "3.12"
include = ["src", "tests"]

[tool.pytest.ini_options]
markers = [
    "integration: tests that hit the real MiniMax API (slow, needs MINIMAX_API_KEY)",
]
addopts = "-m 'not integration'"
```

Note the `pytest` config: it declares the `integration` marker and **excludes integration tests by default** (`addopts = "-m 'not integration'"`). This is what keeps CI fast — the LLM-dependent integration tests skip unless you opt in with `uv run pytest -m integration`.

### 4. Create the package skeleton

```bash
mkdir -p src/safety_rag tests
```

Write `src/safety_rag/__init__.py` with the package entry point:

```python
def main() -> None:
    print("Hello from safety-rag!")
```

This `main()` is what the `[project.scripts]` entry `safety-rag = "safety_rag:main"` resolves to; running `uv run safety-rag` should print the greeting. Later phases will overwrite this with the real CLI.

Write the smoke test:

```python
# tests/test_smoke.py
def test_trivial():
    assert True
```

**Verify:** `uv run pytest tests/ -v` shows `1 passed`. (The full pytest config in step 3 will filter by `not integration` automatically — at this stage the smoke test is the only thing that runs.)

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

**Verify:** `git status` shows only the files you intend to commit, not the ignored ones. The `data/raw/`, `data/processed/`, `evals/results/`, `*.qdrant/` lines are the ones that matter for Phase 1+ — large regenerable artefacts stay out of git.

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

The `ruff format --check` step is what enforces the line-length = 80 / formatting rules from step 3. Drop it if you prefer auto-format on commit.

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
- `uv run safety-rag` prints `Hello from safety-rag!`
- GitHub Actions shows green on `main`

## Commit

The single commit at the end of step 7 is the phase commit. No extra cleanup needed.

## Pitfalls

- **`uv init --package` vs `uv init`** — the `--package` flag creates the `src/safety_rag/` layout (what you want). Without it, you get a flat module.
- **`uv sync --all-extras` vs `uv add --dev <pkg>`** — for a fresh clone the declarative `pyproject.toml` + `uv sync --all-extras` pattern is reproducible. The `uv add` pattern still works for adding one-off deps.
- **Python version pinning** — keep `requires-python = ">=3.12"` and the GitHub Actions matrix pinned to 3.12. Drifting versions will bite you in Phase 2 (`sentence-transformers` and `numpy` ABI).
- **`pyproject.toml` line-length 80, not 100** — the build settled on 80 to keep prompt strings readable in code reviews. If you prefer 100, change it in `pyproject.toml` and re-run `uv run ruff format .`.
- **`.gitignore` for `data/`** — the corpus you'll download in Phase 1 is large and regeneratable. Committing it bloats the repo.
- **Secrets in CI** — don't put your `MINIMAX_API_KEY` in `ci.yml`. Integration tests are skipped by default; if you ever enable them in CI, use `${{ secrets.MINIMAX_API_KEY }}` (Phase 4 will introduce the eval workflow that does this).

## What's next

Phase 1 — Ingestion. You'll download the EUR-Lex corpus from the EU Cellar endpoint, parse it with `selectolax`, and write a structure-aware chunker.
