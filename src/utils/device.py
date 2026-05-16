from __future__ import annotations

import torch


def get_device(prefer: str = "auto") -> torch.device:
    """Select cpu, mps, or cuda with a stable project-wide interface."""

    prefer = prefer.lower()
    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if prefer == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")
    if prefer == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device preference: {prefer}")
