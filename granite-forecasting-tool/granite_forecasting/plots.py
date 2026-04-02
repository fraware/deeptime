"""Plotting helpers for Streamlit."""

from __future__ import annotations

import pandas as pd
import plotly.express as px


def interactive_plot(actual, forecast, title: str = "Forecast vs Actual"):
    """Overlay actual vs forecast as a Plotly line chart (index on x-axis)."""
    actual_arr = _to_1d(actual)
    forecast_arr = _to_1d(forecast)
    m = min(len(actual_arr), len(forecast_arr))
    actual_arr = actual_arr[:m]
    forecast_arr = forecast_arr[:m]
    df = pd.DataFrame(
        {"Time": range(m), "Actual": actual_arr, "Forecast": forecast_arr}
    )
    return px.line(df, x="Time", y=["Actual", "Forecast"], title=title)


def interactive_plot_rolling(
    actual_ctx,
    baseline_forecast,
    extended_forecast,
    context_length: int,
    title: str = "Baseline vs extended recursive forecast",
):
    """Plot context tail, native-length baseline, and full recursive horizon."""
    ctx = _to_1d(actual_ctx)
    base = _to_1d(baseline_forecast)
    ext = _to_1d(extended_forecast)
    show_ctx = min(context_length, len(ctx))
    tail = ctx[-show_ctx:]
    t0 = list(range(-show_ctx, 0))
    rows = [{"Time": t, "Series": "History", "Value": float(tail[i])} for i, t in enumerate(t0)]
    for i in range(len(base)):
        rows.append({"Time": i, "Series": "Baseline (native horizon)", "Value": float(base[i])})
    for i in range(len(ext)):
        rows.append({"Time": i, "Series": "Recursive (full horizon)", "Value": float(ext[i])})
    df = pd.DataFrame(rows)
    return px.line(df, x="Time", y="Value", color="Series", title=title)


def _to_1d(x):
    import numpy as np

    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x).reshape(-1)
    return arr
