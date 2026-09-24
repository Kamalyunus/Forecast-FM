"""Torch device and dtype selection: Apple Silicon (MPS) first-class.

`auto` picks CUDA, then MPS, then CPU. On MPS:
- float32 is the default. bfloat16 needs macOS 14+ and a recent torch;
  check accuracy parity on the sample before adopting it.
- PYTORCH_ENABLE_MPS_FALLBACK=1 is set, so the rare op MPS lacks runs on
  CPU instead of raising.
"""

from __future__ import annotations

import os

DTYPES = ("float32", "bfloat16", "float16")


def resolve_device(device: str = "auto") -> str:
    import torch

    if device != "auto":
        if device.startswith("mps") and not torch.backends.mps.is_available():
            raise RuntimeError("device 'mps' requested but MPS is not available "
                               "(needs Apple Silicon and an arm64 Python/torch build)")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("device 'cuda' requested but CUDA is not available")
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(name: str, device: str):
    import torch

    if name not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}, got {name!r}")
    if name == "float16" and device == "cpu":
        raise ValueError("float16 is not supported on CPU; use float32 or bfloat16")
    return getattr(torch, name)


def prepare(device: str) -> None:
    if device.startswith("mps"):
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def empty_cache(device: str) -> None:
    if device == "cpu":
        return
    import torch

    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    elif device.startswith("mps"):
        torch.mps.empty_cache()


def peak_memory_mb(device: str) -> float | None:
    """Peak accelerator memory for CUDA; current driver allocation for MPS
    (MPS has no peak counter); None on CPU."""
    if device == "cpu":
        return None
    import torch

    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated() / 2**20
    if device.startswith("mps"):
        return torch.mps.driver_allocated_memory() / 2**20
    return None


def describe() -> dict:
    """Environment facts for `env` and run records."""
    import platform

    info = {"python": platform.python_version(), "machine": platform.machine(),
            "platform": platform.platform()}
    try:
        import torch

        info.update(torch=torch.__version__, cuda=torch.cuda.is_available(),
                    mps=torch.backends.mps.is_available(), auto_device=resolve_device("auto"))
    except ImportError:
        info["torch"] = None
    try:
        from importlib.metadata import version

        info["chronos_forecasting"] = version("chronos-forecasting")
    except Exception:
        info["chronos_forecasting"] = None
    return info
