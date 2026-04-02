# Granite forecasting tool (application package)

This directory is the **runnable application** and **installable Python package** for DeepTime.

## Quick start

```bash
# From this directory
pip install -r requirements.txt
streamlit run app.py
```

Editable install with dev dependencies:

```bash
pip install -e ".[dev]"
```

Python **3.11+** required (see [`pyproject.toml`](pyproject.toml)).

## What lives here

| Item | Role |
|------|------|
| [`app.py`](app.py) | Streamlit entrypoint (UI only) |
| [`granite_forecasting/`](granite_forecasting/) | Package modules (forecasting logic, data I/O, plots) |
| [`pyproject.toml`](pyproject.toml) | Metadata, dependencies, Ruff, pytest config |
| [`requirements.txt`](requirements.txt) | Pip-installable list (matches runtime deps; pinned `granite-tsfm` Git revision) |
| [`Dockerfile`](Dockerfile) | Multi-stage image (Python 3.12, non-root user, health check) |
| [`.dockerignore`](.dockerignore) | Keeps test artifacts and caches out of the image |
| [`tests/`](tests/) | Pytest suite |
| [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example) | Template for Hugging Face Hub token (copy to `secrets.toml`, do not commit) |

### Package modules (`granite_forecasting`)

| Module | Responsibility |
|--------|----------------|
| `config.py` | Constants, default model IDs, dataset URLs, narrow warning filters, `OUT_DIR` |
| `data_io.py` | Timestamp detection/parsing, monotonicity checks, M4 wide-to-long helper |
| `plots.py` | Plotly helpers (baseline vs recursive horizon) |
| `rolling.py` | `RecursivePredictor` integration for extended horizons |
| `zero_shot.py` | Zero-shot pipeline and cached model load |
| `channel_mix.py` | Bike-sharing channel-mix fine-tuning demo |
| `m4_hourly.py` | M4 Hourly-train loading and TTM v1 evaluation |

## Docker (from repository root)

```bash
docker build -f granite-forecasting-tool/Dockerfile -t deeptime-forecast granite-forecasting-tool
docker run --rm -p 8501:8501 deeptime-forecast
```

Build-arg **`STREAMLIT_ENABLE_XSRF`** (default `true`): set to `false` if your host (e.g. some Hugging Face Spaces) breaks with XSRF protection enabled.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | Hugging Face Hub authentication for model download (also configurable via Streamlit secrets) |
| `STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION` | Set by the Dockerfile from `STREAMLIT_ENABLE_XSRF` build arg |

## Outputs

Evaluation and plotting may write under **`dashboard_outputs/`** (created at runtime). Add it to `.gitignore` locally if you generate large artifacts (root `.gitignore` already ignores common patterns).

## Full documentation

See the [repository README](../README.md) for architecture, deployment matrix, parameters, data format, and CI overview.
