"""M4 hourly evaluation using official M4 training data reshaped for the preprocessor."""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import streamlit as st
import torch
from transformers import Trainer, TrainingArguments
from tsfm_public import TimeSeriesPreprocessor, get_datasets
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
from tsfm_public.toolkit.visualization import plot_predictions

from granite_forecasting import config
from granite_forecasting.data_io import m4_hourly_wide_row_to_frame


def load_m4_hourly_wide(url: str | None = None) -> pd.DataFrame:
    """Load the M4 Hourly-train.csv wide matrix from the Monash M4-methods repository."""
    src = url or config.M4_HOURLY_TRAIN_URL
    return pd.read_csv(src)


def run_m4_hourly_example(series_row_index: int = 0, use_local_fixture: bool = False):
    st.write("## M4 Hourly Example")
    st.info(
        "Loads the official M4 **Hourly-train** matrix, selects one series row, "
        "and converts it to long format (`date`, `target`) for Granite TTM v1."
    )
    try:
        if use_local_fixture:
            import pathlib

            root = pathlib.Path(__file__).resolve().parents[1]
            fixture = root / "tests" / "fixtures" / "m4_hourly_sample.csv"
            wide = pd.read_csv(fixture)
        else:
            wide = load_m4_hourly_wide()
    except Exception as e:
        st.error("Could not load M4 hourly dataset: " + str(e))
        return

    max_row = len(wide) - 1
    if series_row_index > max_row:
        series_row_index = 0
        st.warning(f"Series index out of range; using row 0 (valid range 0–{max_row}).")

    try:
        m4_data = m4_hourly_wide_row_to_frame(wide, series_row_index)
    except Exception as e:
        st.error("Failed to reshape M4 series: " + str(e))
        return

    st.write("### M4 Hourly Series Preview (long format)")
    st.dataframe(m4_data.head())
    context_length = 512
    forecast_length = 48
    timestamp_column = "date"
    id_columns: list = []
    target_columns = ["target"]

    n = len(m4_data)
    if n < context_length + forecast_length + 10:
        st.warning(
            f"Series length {n} is short for context {context_length}; "
            "consider a different series index."
        )

    split_config = {
        "train": [0, int(n * 0.7)],
        "valid": [int(n * 0.7), int(n * 0.85)],
        "test": [int(n * 0.85), n],
    }
    column_specifiers = {
        "timestamp_column": timestamp_column,
        "id_columns": id_columns,
        "target_columns": target_columns,
        "control_columns": [],
    }
    tsp = TimeSeriesPreprocessor(
        **column_specifiers,
        context_length=context_length,
        prediction_length=forecast_length,
        scaling=True,
        encode_categorical=False,
        scaler_type="standard",
    )
    _dtrain, _dvalid, dset_test = get_datasets(tsp, m4_data, split_config)
    st.write("Data split completed.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyTimeMixerForPrediction.from_pretrained(
        config.TTM_MODEL_PATH_M4,
        revision="main",
        prediction_filter_length=forecast_length,
    ).to(device)
    st.write("Running zero-shot evaluation on M4 hourly (selected series)...")
    temp_dir = tempfile.mkdtemp()
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=temp_dir,
            per_device_eval_batch_size=64,
            report_to="none",
        ),
    )
    eval_output = trainer.evaluate(dset_test)
    st.write("Zero-shot evaluation metrics on M4 hourly:")
    st.json(eval_output)
    plot_dir = os.path.join(config.OUT_DIR, "m4_hourly", "zero_shot")
    os.makedirs(plot_dir, exist_ok=True)
    try:
        plot_predictions(
            model=trainer.model,
            dset=dset_test,
            plot_dir=plot_dir,
            plot_prefix="m4_zero_shot",
            indices=[0],
            channel=0,
        )
    except Exception as e:
        st.error("Error plotting M4 zero-shot predictions: " + str(e))
        return
    for file in os.listdir(plot_dir):
        if file.endswith(".png"):
            st.image(os.path.join(plot_dir, file), caption=file)
    st.caption(
        "Fine-tuning on M4 can be added similarly using the same long-format dataframe."
    )
