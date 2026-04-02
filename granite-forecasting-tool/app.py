"""
Streamlit entrypoint for the Granite time-series forecasting dashboard.

Run from this directory: `streamlit run app.py`
"""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from granite_forecasting import config
from granite_forecasting.channel_mix import run_channel_mix_finetuning
from granite_forecasting.data_io import (
    default_timestamp_column,
    detect_datetime_columns,
    parse_timestamp_column,
    validate_monotonic_timestamps,
)
from granite_forecasting.m4_hourly import run_m4_hourly_example
from granite_forecasting.zero_shot import run_zero_shot_forecasting


def main():
    st.title("Interactive Time-Series Forecasting Dashboard")
    st.markdown(
        """
        Run forecasting experiments with IBM **Granite** time-series models (TTM).
        Choose a mode below:
        - **Zero-shot Evaluation**
        - **Channel-Mix Finetuning Example**
        - **M4 Hourly Example**
        """
    )

    mode = st.selectbox(
        "Select Evaluation Mode",
        options=[
            "Zero-shot Evaluation",
            "Channel-Mix Finetuning Example",
            "M4 Hourly Example",
        ],
    )

    if mode == "Zero-shot Evaluation":
        dataset_source = st.radio(
            "Dataset Source", options=["Default (ETTh1)", "Upload CSV"]
        )
        timestamp_column: str

        if dataset_source == "Default (ETTh1)":
            dataset_path = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
            try:
                data = pd.read_csv(dataset_path, parse_dates=["date"])
            except Exception:
                st.error("Error loading default dataset.")
                return
            timestamp_column = "date"
            st.write("### Default Dataset Preview")
            st.dataframe(data.head())
            selected_target_columns = [
                "HUFL",
                "HULL",
                "MUFL",
                "MULL",
                "LUFL",
                "LULL",
                "OT",
            ]
        else:
            uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
            if not uploaded_file:
                st.info("Awaiting CSV file upload.")
                return
            raw = uploaded_file.read()
            try:
                data = pd.read_csv(io.BytesIO(raw))
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                return
            st.write("### Uploaded Data Preview")
            st.dataframe(data.head())

            dt_cols = detect_datetime_columns(data)
            default_ts = default_timestamp_column(data, dt_cols)
            if not dt_cols and default_ts is None:
                st.error(
                    "No datetime-like column found. Add a parseable date/timestamp column."
                )
                return
            options = dt_cols if dt_cols else [default_ts]
            timestamp_column = st.selectbox(
                "Timestamp column",
                options=options,
                index=options.index(default_ts) if default_ts in options else 0,
            )
            try:
                data = parse_timestamp_column(data, timestamp_column)
                validate_monotonic_timestamps(data[timestamp_column], strict=False)
            except (KeyError, ValueError) as e:
                st.error(str(e))
                return

            available_columns = [
                col for col in data.columns if col != timestamp_column
            ]
            selected_target_columns = st.multiselect(
                "Select Target Column(s)",
                options=available_columns,
                default=available_columns,
            )

        if dataset_source == "Default (ETTh1)":
            available_exog = [
                col
                for col in data.columns
                if col not in ([timestamp_column] + selected_target_columns)
            ]
        else:
            available_exog = [
                col
                for col in data.columns
                if col not in ([timestamp_column] + selected_target_columns)
            ]

        selected_conditional_columns = st.multiselect(
            "Select Exogenous/Control Columns", options=available_exog, default=[]
        )
        rolling_extension = st.number_input(
            "Rolling Forecast Extension (Extra Steps)", value=0, min_value=0, step=1
        )
        forecast_index = st.slider(
            "Select Forecast Index for Plotting",
            min_value=0,
            max_value=max(len(data) - 1, 0),
            value=0,
        )
        context_length = st.number_input(
            "Context Length", value=config.DEFAULT_CONTEXT_LENGTH, step=64
        )
        prediction_length = st.number_input(
            "Prediction Length", value=config.DEFAULT_PREDICTION_LENGTH, step=1
        )
        batch_size = st.number_input("Batch Size", value=64, step=1)
        if st.button("Run Zero-shot Evaluation"):
            with st.spinner("Running zero-shot evaluation..."):
                run_zero_shot_forecasting(
                    data,
                    timestamp_column,
                    context_length,
                    prediction_length,
                    batch_size,
                    selected_target_columns,
                    selected_conditional_columns,
                    rolling_extension,
                    forecast_index,
                )

    elif mode == "Channel-Mix Finetuning Example":
        if st.button("Run Channel-Mix Finetuning Example"):
            with st.spinner("Running channel-mix finetuning..."):
                run_channel_mix_finetuning()

    elif mode == "M4 Hourly Example":
        use_fixture = st.checkbox(
            "Use bundled tiny CSV fixture (offline / CI demo)", value=False
        )
        series_idx = st.number_input(
            "M4 series row index (from Hourly-train.csv)", min_value=0, value=0, step=1
        )
        if st.button("Run M4 Hourly Example"):
            with st.spinner("Running M4 hourly example..."):
                run_m4_hourly_example(
                    series_row_index=int(series_idx),
                    use_local_fixture=use_fixture,
                )


if __name__ == "__main__":
    main()
