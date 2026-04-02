"""CSV loading, timestamp column detection, and M4 hourly reshaping."""

from __future__ import annotations

import pandas as pd


def detect_datetime_columns(df: pd.DataFrame) -> list[str]:
    """Return column names that are datetime-like (dtype or parseable as datetimes)."""
    found: list[str] = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            found.append(str(col))
            continue
        if df[col].dtype == object:
            sample = df[col].dropna().head(50)
            if sample.empty:
                continue
            try:
                parsed = pd.to_datetime(sample, errors="raise")
                if parsed.notna().mean() >= 0.8:
                    found.append(str(col))
            except (ValueError, TypeError):
                continue
    return found


def default_timestamp_column(df: pd.DataFrame, candidates: list[str] | None = None) -> str | None:
    """Pick a sensible default timestamp column name."""
    cols = candidates if candidates is not None else detect_datetime_columns(df)
    if not cols:
        for name in ("date", "timestamp", "time", "datetime", "ds"):
            if name in df.columns:
                return name
        return None
    priority = ("date", "timestamp", "time", "datetime", "ds")
    for p in priority:
        if p in cols:
            return p
    return cols[0]


def parse_timestamp_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return a copy with `column` coerced to datetime64."""
    out = df.copy()
    if column not in out.columns:
        raise KeyError(f"Timestamp column {column!r} not in dataframe columns.")
    out[column] = pd.to_datetime(out[column], errors="coerce")
    if out[column].isna().all():
        raise ValueError(f"Could not parse any valid datetimes in column {column!r}.")
    return out


def validate_monotonic_timestamps(series: pd.Series, strict: bool = False) -> None:
    """Raise if timestamps are not non-decreasing."""
    if series.isna().any():
        raise ValueError("Timestamp column contains NaT after parsing.")
    diffs = series.diff().dropna()
    if strict and not (diffs > pd.Timedelta(0)).all():
        raise ValueError("Timestamps are not strictly increasing.")
    if not strict and (diffs < pd.Timedelta(0)).any():
        raise ValueError("Timestamps are not sorted (found decreasing values).")


def m4_hourly_wide_row_to_frame(
    wide_df: pd.DataFrame,
    row_index: int,
    *,
    freq: str = "h",
    start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Convert one row of M4 Hourly-train.csv (series id in col 0, V1..Vn observations) to long format.

    M4 hourly files use the first column for the series identifier and remaining columns for
    observations; trailing cells may be empty.
    """
    if row_index < 0 or row_index >= len(wide_df):
        msg = f"row_index {row_index} out of range (len={len(wide_df)})."
        raise IndexError(msg)
    row = wide_df.iloc[row_index]
    values = pd.to_numeric(row.iloc[1:], errors="coerce").dropna()
    if values.empty:
        raise ValueError("No numeric observations in selected M4 series row.")
    n = len(values)
    t0 = start if start is not None else pd.Timestamp("2000-01-01")
    dates = pd.date_range(start=t0, periods=n, freq=freq)
    return pd.DataFrame({"date": dates, "target": values.to_numpy(dtype=float)})


def load_csv_bytes(content: bytes, parse_dates: list[str] | None = None) -> pd.DataFrame:
    """Load CSV from raw bytes (e.g. Streamlit upload)."""
    import io

    if parse_dates:
        return pd.read_csv(io.BytesIO(content), parse_dates=parse_dates)
    return pd.read_csv(io.BytesIO(content))
