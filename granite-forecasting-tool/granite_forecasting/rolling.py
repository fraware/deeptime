"""Extended-horizon forecasts using IBM TSFM RecursivePredictor."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def tensor_batch_from_dataset_sample(
    sample,
    device: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Build (past_values, past_observed_mask) batch tensors from one dataset example."""
    if not isinstance(sample, dict):
        return None, None
    past = sample.get("past_values")
    if past is None:
        past = sample.get("past_target")
    if past is None:
        return None, None
    if not isinstance(past, torch.Tensor):
        past = torch.as_tensor(past, dtype=torch.float32)
    if past.dim() == 2:
        past = past.unsqueeze(0)
    past = past.to(device)
    mask = sample.get("past_observed_mask")
    if mask is not None:
        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask, dtype=torch.bool)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        mask = mask.to(device)
    return past, mask


def recursive_forecast_batch(
    base_model: torch.nn.Module,
    *,
    past_values: torch.Tensor,
    past_observed_mask: torch.Tensor | None,
    requested_prediction_length: int,
    model_prediction_length: int,
    device: str,
) -> torch.Tensor | None:
    """
    Run RecursivePredictor for a longer horizon than the base model's native prediction length.

    Returns tensor of shape (batch, requested_prediction_length, channels) or None on failure.
    """
    try:
        from tsfm_public.toolkit.recursive_predictor import (
            RecursivePredictor,
            RecursivePredictorConfig,
        )
    except ImportError as e:
        logger.warning("RecursivePredictor import failed: %s", e)
        return None

    try:
        cfg = RecursivePredictorConfig(
            model=base_model,
            requested_prediction_length=requested_prediction_length,
            model_prediction_length=model_prediction_length,
            loss="mse",
        )
        wrapper = RecursivePredictor(cfg).to(device)
        wrapper.eval()
        with torch.no_grad():
            out = wrapper(
                past_values=past_values,
                future_values=None,
                past_observed_mask=past_observed_mask,
                future_observed_mask=None,
                return_dict=True,
            )
        return out.prediction_outputs
    except Exception as e:
        logger.exception("Recursive forecast failed: %s", e)
        return None
