"""Model registry: experiment configs name a family; params go to its
constructor."""

from __future__ import annotations

from .base import Forecaster
from .baselines import Croston, Naive, SeasonalNaive
from .chronos2 import Chronos2

REGISTRY: dict[str, type[Forecaster]] = {
    "naive": Naive,
    "seasonal_naive": SeasonalNaive,
    "croston": Croston,
    "chronos2": Chronos2,
}


def create_model(name: str, params: dict | None = None) -> Forecaster:
    if name not in REGISTRY:
        raise ValueError(f"unknown model {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name](params)
