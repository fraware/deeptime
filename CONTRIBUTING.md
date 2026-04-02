# Contributing to DeepTime

Thank you for your interest in contributing. We welcome issues, documentation improvements, and pull requests.

## Code of conduct

This project adheres to the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you agree to abide by its terms.

## Ways to contribute

### Reporting bugs

Open a GitHub issue with:

- Steps to reproduce
- Expected vs actual behavior
- Logs or screenshots if helpful
- OS, Python version, and whether you use Docker or local venv

### Suggesting enhancements

Open an issue with a clear goal, proposed UX or API, and constraints (e.g. CPU-only, HF Spaces).

### Pull requests

1. **Fork** the repository on GitHub and **clone your fork**:

   ```bash
   git clone https://github.com/<your-username>/deeptime.git
   cd deeptime
   ```

2. Add `upstream` if you like:

   ```bash
   git remote add upstream https://github.com/fraware/deeptime.git
   ```

3. Create a focused branch: `feature/...`, `fix/...`, or `docs/...`.

4. Make changes; add or update **tests** under [`granite-forecasting-tool/tests/`](granite-forecasting-tool/tests/) when behavior changes.

5. Run **Ruff** and **pytest** locally (see below).

6. Open a PR against `main` with a concise description; **CI** must pass (lint, tests, Docker build).

## Development setup

Requirements: **Python 3.11+**, **git** (for installing `granite-tsfm` from Git), enough disk for PyTorch and model caches.

```bash
cd granite-forecasting-tool
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows
pip install --upgrade pip
pip install -e ".[dev]"
```

The `[dev]` extra installs **Ruff**, **pytest**, and **pytest-cov** (see [`pyproject.toml`](granite-forecasting-tool/pyproject.toml)).

**Alternative (runtime only):**

```bash
pip install -r requirements.txt
```

Runtime dependencies and the **pinned `granite-tsfm` Git revision** are listed in [`requirements.txt`](granite-forecasting-tool/requirements.txt).

### Optional: stricter lockfiles

For fully pinned transitive dependencies you can use **`uv`** (`uv lock`) or **`pip-tools`** (`pip-compile`) on a `requirements.in`; this repo currently relies on minimum versions in `pyproject.toml` plus a **fixed `granite-tsfm` commit**. Document any new lockfile in the PR.

### Optional: pre-commit

From the **repository root** (not only `granite-forecasting-tool/`):

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Hooks use Ruff with [`granite-forecasting-tool/pyproject.toml`](granite-forecasting-tool/pyproject.toml).

## Checks before you push

From **`granite-forecasting-tool/`**:

```bash
ruff check .
pytest
```

**Coverage (optional):**

```bash
pytest --cov=granite_forecasting --cov-report=term-missing
```

## CI expectations

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on **Python 3.11 and 3.12**:

1. `pip install -e ".[dev]"` from `granite-forecasting-tool/`
2. `ruff check .`
3. `pytest`
4. `docker build -f granite-forecasting-tool/Dockerfile granite-forecasting-tool` from the repo root

[Dependabot](.github/dependabot.yml) opens weekly PRs for GitHub Actions and pip dependencies under `granite-forecasting-tool/`.

## Project structure (for contributors)

| Area | Location |
|------|----------|
| Streamlit entry | [`granite-forecasting-tool/app.py`](granite-forecasting-tool/app.py) |
| Package | [`granite-forecasting-tool/granite_forecasting/`](granite-forecasting-tool/granite_forecasting/) |
| Tests | [`granite-forecasting-tool/tests/`](granite-forecasting-tool/tests/) |
| Docker | [`granite-forecasting-tool/Dockerfile`](granite-forecasting-tool/Dockerfile), [`.dockerignore`](granite-forecasting-tool/.dockerignore) |
| User docs | Root [`README.md`](README.md) |

## Code style

- Follow **PEP 8**; **Ruff** is the enforced linter (`E`, `F`, `I`, `W`, line length 100).
- Prefer small, reviewable commits.
- Use imperative commit subjects (e.g. `Fix M4 loader bounds check`).
- Do **not** commit secrets. For local Streamlit secrets, copy [`granite-forecasting-tool/.streamlit/secrets.toml.example`](granite-forecasting-tool/.streamlit/secrets.toml.example) to `granite-forecasting-tool/.streamlit/secrets.toml` (that file is gitignored).

## Questions

Open an issue for discussion.

Thank you for helping improve DeepTime.
