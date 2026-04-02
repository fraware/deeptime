"""Channel-mix fine-tuning example (bike sharing) for Streamlit."""

from __future__ import annotations

import math
import os

import pandas as pd
import streamlit as st
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
from transformers.integrations import INTEGRATION_TO_CALLBACK
from tsfm_public import (
    TimeSeriesPreprocessor,
    TrackingCallback,
    count_parameters,
    get_datasets,
)
from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.visualization import plot_predictions

from granite_forecasting import config


def _training_args_with_eval_epoch(**kwargs):
    """Support both `evaluation_strategy` and `eval_strategy` across transformers versions."""
    try:
        return TrainingArguments(evaluation_strategy="epoch", **kwargs)
    except TypeError:
        return TrainingArguments(eval_strategy="epoch", **kwargs)


@st.cache_resource(show_spinner="Loading channel-mix base model…")
def load_channel_mix_model(
    context_length: int,
    forecast_length: int,
    num_input_channels: int,
    prediction_channel_indices,
):
    return get_model(
        config.TTM_MODEL_PATH_CHANNEL_MIX,
        context_length=context_length,
        prediction_length=forecast_length,
        num_input_channels=num_input_channels,
        decoder_mode="mix_channel",
        prediction_channel_indices=prediction_channel_indices,
    )


def run_channel_mix_finetuning():
    st.write("## Channel-Mix Finetuning Example (Bike Sharing Data)")
    st.caption(
        f"Data is loaded from a third-party URL; availability may change. "
        f"URL: `{config.BIKE_SHARING_CSV_URL}`"
    )
    target_dataset = "bike_sharing"
    timestamp_column = "dteday"
    id_columns: list = []
    try:
        data = pd.read_csv(
            config.BIKE_SHARING_CSV_URL, parse_dates=[timestamp_column]
        )
    except Exception as e:
        st.error("Error loading bike sharing dataset: " + str(e))
        return
    data[timestamp_column] = pd.to_datetime(data[timestamp_column])
    data[timestamp_column] = data[timestamp_column] + pd.to_timedelta(
        data.groupby(data[timestamp_column].dt.date).cumcount(), unit="h"
    )
    st.write("### Bike Sharing Data Preview")
    st.dataframe(data.head())

    column_specifiers = {
        "timestamp_column": timestamp_column,
        "id_columns": id_columns,
        "target_columns": ["casual", "registered", "cnt"],
        "conditional_columns": [
            "season",
            "yr",
            "mnth",
            "holiday",
            "weekday",
            "workingday",
            "weathersit",
            "temp",
            "atemp",
            "hum",
            "windspeed",
        ],
    }
    n = len(data)
    split_config = {
        "train": [0, int(n * 0.5)],
        "valid": [int(n * 0.5), int(n * 0.75)],
        "test": [int(n * 0.75), n],
    }
    context_length = 512
    forecast_length = 96

    tsp = TimeSeriesPreprocessor(
        **column_specifiers,
        context_length=context_length,
        prediction_length=forecast_length,
        scaling=True,
        encode_categorical=False,
        scaler_type="standard",
    )
    train_dataset, valid_dataset, test_dataset = get_datasets(tsp, data, split_config)
    st.write("Data split completed.")

    finetune_forecast_model = load_channel_mix_model(
        context_length,
        forecast_length,
        tsp.num_input_channels,
        tsp.prediction_channel_indices,
    )
    st.write(
        "Number of params before freezing backbone:",
        count_parameters(finetune_forecast_model),
    )
    for param in finetune_forecast_model.backbone.parameters():
        param.requires_grad = False
    st.write(
        "Number of params after freezing backbone:",
        count_parameters(finetune_forecast_model),
    )

    num_epochs = 50
    batch_size = 64
    learning_rate = 0.001
    optimizer = AdamW(finetune_forecast_model.parameters(), lr=learning_rate)
    scheduler = OneCycleLR(
        optimizer,
        learning_rate,
        epochs=num_epochs,
        steps_per_epoch=math.ceil(len(train_dataset) / batch_size),
    )
    out_dir = os.path.join(config.OUT_DIR, target_dataset)
    os.makedirs(out_dir, exist_ok=True)
    finetune_args = _training_args_with_eval_epoch(
        output_dir=os.path.join(out_dir, "output"),
        overwrite_output_dir=True,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        do_eval=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=0,
        report_to="none",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        logging_dir=os.path.join(out_dir, "logs"),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=config.SEED,
    )
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=10,
        early_stopping_threshold=1e-5,
    )
    tracking_callback = TrackingCallback()
    finetune_trainer = Trainer(
        model=finetune_forecast_model,
        args=finetune_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[early_stopping_callback, tracking_callback],
        optimizers=(optimizer, scheduler),
    )
    cb = INTEGRATION_TO_CALLBACK.get("codecarbon")
    if cb is not None:
        try:
            finetune_trainer.remove_callback(cb)
        except Exception:
            pass
    st.write("Starting channel-mix finetuning...")
    finetune_trainer.train()
    st.write("Evaluating finetuned model on test set...")
    eval_output = finetune_trainer.evaluate(test_dataset)
    st.write("Few-shot (channel-mix) evaluation metrics:")
    st.json(eval_output)
    plot_dir = os.path.join(out_dir, "channel_mix_plots")
    os.makedirs(plot_dir, exist_ok=True)
    try:
        plot_predictions(
            model=finetune_trainer.model,
            dset=test_dataset,
            plot_dir=plot_dir,
            plot_prefix="test_channel_mix",
            indices=[0],
            channel=0,
        )
    except Exception as e:
        st.error("Error plotting channel mix predictions: " + str(e))
        return
    for file in os.listdir(plot_dir):
        if file.endswith(".png"):
            st.image(os.path.join(plot_dir, file), caption=file)
