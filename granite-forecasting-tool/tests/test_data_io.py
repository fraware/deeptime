"""Tests for data_io helpers (no heavy ML dependencies)."""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from granite_forecasting.data_io import (
    default_timestamp_column,
    detect_datetime_columns,
    load_csv_bytes,
    m4_hourly_wide_row_to_frame,
    parse_timestamp_column,
    validate_monotonic_timestamps,
)


def test_detect_datetime_columns():
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "x": [1.0, 2.0],
        }
    )
    assert "date" in detect_datetime_columns(df)


def test_parse_timestamp_column():
    df = pd.DataFrame({"ts": ["2021-06-01", "2021-06-02"]})
    out = parse_timestamp_column(df, "ts")
    assert pd.api.types.is_datetime64_any_dtype(out["ts"])


def test_validate_monotonic_timestamps():
    s = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    validate_monotonic_timestamps(s, strict=False)
    with pytest.raises(ValueError):
        validate_monotonic_timestamps(
            pd.to_datetime(["2020-01-03", "2020-01-02"]), strict=False
        )


def test_default_timestamp_column():
    df = pd.DataFrame({"date": ["2020-01-01"], "v": [1]})
    df["date"] = pd.to_datetime(df["date"])
    assert default_timestamp_column(df) == "date"


def test_m4_hourly_wide_row_to_frame():
    root = pathlib.Path(__file__).resolve().parents[1]
    wide = pd.read_csv(root / "tests" / "fixtures" / "m4_hourly_sample.csv")
    long_df = m4_hourly_wide_row_to_frame(wide, 0)
    assert list(long_df.columns) == ["date", "target"]
    assert len(long_df) == 7
    assert long_df["target"].iloc[-1] == pytest.approx(4.0)


def test_load_csv_bytes():
    raw = b"a,b\n1,2\n"
    df = load_csv_bytes(raw)
    assert list(df.columns) == ["a", "b"]
