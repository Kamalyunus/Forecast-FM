"""Forecast-FM: Chronos-2 demand forecasting for replenishment."""

import os

# must be set before torch is first imported: torch registers the MPS CPU
# fallback at import time, so setting it later has no effect
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

__version__ = "0.2.0"
