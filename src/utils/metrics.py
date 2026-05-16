from __future__ import annotations

import torch


def regression_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    prediction = prediction.detach().cpu()
    target = target.detach().cpu()
    error = prediction - target
    mse = torch.mean(error**2).item()
    mae = torch.mean(torch.abs(error)).item()
    max_absolute_error = torch.max(torch.abs(error)).item()
    l2_relative_error = (torch.linalg.norm(error) / torch.linalg.norm(target)).item()
    return {
        "l2_relative_error": l2_relative_error,
        "mse": mse,
        "mae": mae,
        "max_absolute_error": max_absolute_error,
    }
