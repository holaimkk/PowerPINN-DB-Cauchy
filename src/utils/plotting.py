from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "powerpinn_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def save_loss_curve(loss_history: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.semilogy(loss_history["epoch"], loss_history["total_loss"], label="total")
    plt.semilogy(loss_history["epoch"], loss_history["physics_loss"], label="physics")
    plt.semilogy(
        loss_history["epoch"],
        loss_history["initial_condition_loss"],
        label="initial condition",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_prediction_plot(predictions: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(predictions["t"], predictions["u_true"], label="truth", linewidth=2)
    plt.plot(predictions["t"], predictions["u_pred"], "--", label="PINN prediction")
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
