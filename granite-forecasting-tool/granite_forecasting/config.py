"""Application constants and defaults."""

import os
import warnings

from transformers import set_seed

# Narrow noisy warnings; avoid blanket ignore

warnings.filterwarnings(
    "ignore",
    message=".*torch.utils._pytree._register_pytree_node.*",
    category=FutureWarning,
)

SEED = 42
set_seed(SEED)

TTM_MODEL_PATH = "ibm-granite/granite-timeseries-ttm-r2"
TTM_MODEL_PATH_CHANNEL_MIX = "ibm-granite/granite-timeseries-ttm-r1"
TTM_MODEL_PATH_M4 = "ibm-granite/granite-timeseries-ttm-v1"
DEFAULT_CONTEXT_LENGTH = 512
DEFAULT_PREDICTION_LENGTH = 96
OUT_DIR = "dashboard_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Pinned third-party dataset URLs (reproducibility)
BIKE_SHARING_CSV_URL = (
    "https://raw.githubusercontent.com/blobibob/bike-sharing-dataset/main/hour.csv"
)
M4_HOURLY_TRAIN_URL = (
    "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/Hourly-train.csv"
)
