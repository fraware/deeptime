"""Zero-shot forecasting pipeline and Streamlit UI helpers."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import streamlit as st
import torch
from transformers import Trainer, TrainingArguments
from tsfm_public import TimeSeriesPreprocessor, get_datasets
from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.visualization import plot_predictions

from granite_forecasting import config
from granite_forecasting.plots import interactive_plot, interactive_plot_rolling
from granite_forecasting.rolling import recursive_forecast_batch, tensor_batch_from_dataset_sample


@st.cache_resource(show_spinner="Loading pretrained model…")
def load_zero_shot_model(
    model_path: str,
    context_length: int,
    prediction_length: int,
):
    return get_model(
        model_path,
        context_length=context_length,
        prediction_length=prediction_length,
    )


def run_zero_shot_forecasting(
    data,
    timestamp_column: str,
    context_length: int,
    prediction_length: int,
    batch_size: int,
    selected_target_columns: list,
    selected_conditional_columns: list,
    rolling_forecast_extension: int,
    selected_forecast_index: int,
):
    st.write("### Preparing Data for Forecasting")
    id_columns: list = []

    if not selected_target_columns:
        target_columns = [col for col in data.columns if col != timestamp_column]
    else:
        target_columns = selected_target_columns

    conditional_columns = selected_conditional_columns

    column_specifiers = {
        "timestamp_column": timestamp_column,
        "id_columns": id_columns,
        "target_columns": target_columns,
        "control_columns": conditional_columns,
    }

    n = len(data)
    split_config = {
        "train": [0, int(n * 0.7)],
        "valid": [int(n * 0.7), int(n * 0.8)],
        "test": [int(n * 0.8), n],
    }

    tsp = TimeSeriesPreprocessor(
        **column_specifiers,
        context_length=context_length,
        prediction_length=prediction_length,
        scaling=True,
        encode_categorical=False,
        scaler_type="standard",
    )
    _dset_train, _dset_valid, dset_test = get_datasets(tsp, data, split_config)
    st.write("Data split into train, validation, and test sets.")

    st.write("### Loading the Pre-trained TTM Model")
    model = load_zero_shot_model(
        config.TTM_MODEL_PATH,
        context_length,
        prediction_length,
    )
    temp_dir = tempfile.mkdtemp()
    training_args = TrainingArguments(
        output_dir=temp_dir,
        per_device_eval_batch_size=batch_size,
        seed=config.SEED,
        report_to="none",
    )
    trainer = Trainer(model=model, args=training_args)

    st.write("### Running Zero-shot Evaluation")
    st.info("Evaluating on the test set...")
    eval_output = trainer.evaluate(dset_test)
    st.write("**Zero-shot Evaluation Metrics:**")
    st.json(eval_output)

    st.write("### Generating Forecast Predictions")
    predictions_dict = trainer.predict(dset_test)
    try:
        predictions_np = predictions_dict.predictions[0]
    except Exception as e:
        st.error("Error extracting predictions: " + str(e))
        return
    st.write("Predictions shape:", predictions_np.shape)

    idx = selected_forecast_index
    if idx >= len(dset_test):
        idx = 0
        st.warning("Forecast index out of range for test set; using index 0.")

    try:
        sample = dset_test[idx]
        actual = (
            sample["target"]
            if isinstance(sample, dict) and "target" in sample
            else dset_test[idx][0]
        )
    except Exception:
        actual = predictions_np[idx]

    baseline_fc = predictions_np[idx]
    fig = interactive_plot(
        actual,
        baseline_fc,
        title=f"Forecast vs Actual for index {idx}",
    )
    st.plotly_chart(fig)

    if rolling_forecast_extension > 0:
        st.write(
            f"### Rolling forecast extension: +{rolling_forecast_extension} steps "
            f"(total horizon {prediction_length + rolling_forecast_extension})"
        )
        st.caption(
            "Recursive extension feeds prior predictions back as context; errors can compound."
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        base_model = trainer.model.to(device)
        sample = dset_test[idx]
        past_values, past_mask = tensor_batch_from_dataset_sample(sample, device)
        if past_values is None:
            st.warning(
                "Could not extract past_values from the dataset sample; skipping recursive plot."
            )
        else:
            total_len = prediction_length + int(rolling_forecast_extension)
            extended = recursive_forecast_batch(
                base_model,
                past_values=past_values,
                past_observed_mask=past_mask,
                requested_prediction_length=total_len,
                model_prediction_length=prediction_length,
                device=device,
            )
            if extended is None:
                st.warning("Recursive prediction failed; see logs for details.")
            else:
                ext_np = extended[0].detach().cpu().numpy()
                base_arr = np.asarray(baseline_fc)
                if base_arr.ndim >= 2:
                    base_1d = base_arr[:, 0].reshape(-1)
                else:
                    base_1d = base_arr.reshape(-1)
                if ext_np.ndim >= 2:
                    ext_1d = ext_np[:, 0].reshape(-1)
                else:
                    ext_1d = ext_np.reshape(-1)
                history = _history_tail_from_sample(sample, actual, channel=0)
                fig2 = interactive_plot_rolling(
                    history,
                    base_1d,
                    ext_1d,
                    context_length=min(context_length, 256),
                    title="History, baseline horizon, and recursive extension",
                )
                st.plotly_chart(fig2)

    plot_dir = os.path.join(config.OUT_DIR, "zero_shot_plots")
    os.makedirs(plot_dir, exist_ok=True)
    try:
        plot_predictions(
            model=trainer.model,
            dset=dset_test,
            plot_dir=plot_dir,
            plot_prefix="test_zeroshot",
            indices=[idx],
            channel=0,
        )
    except Exception as e:
        st.error("Error during static plotting: " + str(e))
        return
    for file in os.listdir(plot_dir):
        if file.endswith(".png"):
            st.image(os.path.join(plot_dir, file), caption=file)


def _history_tail_from_sample(sample, actual, channel: int = 0):
    """Use past_values from a dataset dict for rolling plot context when available."""
    if isinstance(sample, dict) and "past_values" in sample:
        pv = sample["past_values"]
        if hasattr(pv, "detach"):
            pv = pv.detach().cpu().numpy()
        arr = np.asarray(pv, dtype=float)
        if arr.ndim >= 2:
            return arr[:, channel]
        return arr.reshape(-1)
    arr = np.asarray(actual, dtype=float)
    if arr.ndim >= 2:
        return arr[:, channel]
    return arr.reshape(-1)
