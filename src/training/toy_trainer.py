from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from src.losses import compute_toy_pinn_losses
from src.models import MLP
from src.systems import ExponentialDecayODE
from src.utils import get_device, set_seed
from src.utils.metrics import regression_metrics
from src.utils.plotting import save_loss_curve, save_prediction_plot


def _prepare_output_dir(config: dict[str, Any], config_path: Path) -> Path:
    exp_cfg = config["experiment"]
    output_dir = Path(exp_cfg["output_root"]) / exp_cfg["id"]
    if output_dir.exists() and exp_cfg.get("overwrite", False):
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")
    return output_dir


def _setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("toy_pinn")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(output_dir / "train.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def _build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    train_cfg = config["training"]
    name = train_cfg["optimizer"].lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=float(train_cfg["learning_rate"]))
    raise ValueError(f"Unsupported optimizer: {name}")


def _write_experiment_readme(output_dir: Path, config: dict[str, Any], metrics: dict[str, Any]) -> None:
    lines = [
        f"# {config['experiment']['id']} - {config['experiment']['name']}",
        "",
        "本次实验是指数衰减 ODE 的 Toy PINN 运行记录：",
        "",
        "```text",
        "u'(t) = -u(t), u(0) = 1",
        "```",
        "",
        "## 运行命令",
        "",
        "```bash",
        "python scripts/run_toy_pinn.py --config configs/toy_pinn.yaml",
        "```",
        "",
        "## 最终指标",
        "",
    ]
    for key, value in metrics.items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    lines.append("这些数值由本地实验生成。如果本次运行被采纳，应同步登记到外部飞书实验记录表。")
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run_toy_pinn(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(int(config["seed"]))
    device = get_device(config["device"].get("prefer", "auto"))
    output_dir = _prepare_output_dir(config, config_path)
    logger = _setup_logger(output_dir)
    logger.info("Starting toy PINN run")
    logger.info("Using device: %s", device)

    problem_cfg = config["problem"]
    system = ExponentialDecayODE(
        t_min=float(problem_cfg["t_min"]),
        t_max=float(problem_cfg["t_max"]),
        initial_t=float(problem_cfg["initial_t"]),
        initial_u=float(problem_cfg["initial_u"]),
    )

    model_cfg = config["model"]
    model = MLP(
        input_dim=int(model_cfg["input_dim"]),
        output_dim=int(model_cfg["output_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_hidden_layers=int(model_cfg["num_hidden_layers"]),
        activation=str(model_cfg["activation"]),
    ).to(device)
    optimizer = _build_optimizer(model, config)

    t_collocation = system.sample_collocation(int(problem_cfg["n_collocation"]), device)
    loss_cfg = config["loss"]
    train_cfg = config["training"]
    epochs = int(train_cfg["epochs"])
    log_every = int(train_cfg["log_every"])
    loss_history: list[dict[str, float]] = []

    start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        losses = compute_toy_pinn_losses(
            model=model,
            system=system,
            t_collocation=t_collocation,
            physics_weight=float(loss_cfg["physics_weight"]),
            initial_condition_weight=float(loss_cfg["initial_condition_weight"]),
            device=device,
        )
        losses["total_loss"].backward()
        optimizer.step()

        row = {"epoch": epoch}
        row.update({name: value.detach().item() for name, value in losses.items()})
        loss_history.append(row)

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            logger.info(
                "epoch=%d total=%.6e physics=%.6e ic=%.6e",
                epoch,
                row["total_loss"],
                row["physics_loss"],
                row["initial_condition_loss"],
            )

    training_time = time.perf_counter() - start_time

    eval_t = system.eval_grid(int(problem_cfg["n_eval"]), device)
    with torch.no_grad():
        prediction = model(eval_t)
        truth = system.exact_solution(eval_t)
    metrics = regression_metrics(prediction, truth)

    residual_t = system.sample_collocation(int(problem_cfg["n_eval"]), device)
    residual = system.residual(model, residual_t)
    metrics.update(
        {
            "physics_residual_mse": torch.mean(residual**2).detach().cpu().item(),
            "training_time_sec": training_time,
            "epoch": epochs,
            "seed": int(config["seed"]),
            "device": str(device),
        }
    )

    loss_df = pd.DataFrame(loss_history)
    loss_df.to_csv(output_dir / "loss_history.csv", index=False)

    predictions_df = pd.DataFrame(
        {
            "t": eval_t.detach().cpu().reshape(-1).numpy(),
            "u_pred": prediction.detach().cpu().reshape(-1).numpy(),
            "u_true": truth.detach().cpu().reshape(-1).numpy(),
        }
    )
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)

    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)
    torch.save(model.state_dict(), output_dir / "model.pt")
    save_loss_curve(loss_df, output_dir / "figures" / "loss_curve.png")
    save_prediction_plot(predictions_df, output_dir / "figures" / "prediction_vs_truth.png")
    _write_experiment_readme(output_dir, config, metrics)

    logger.info("Finished toy PINN run in %.3f seconds", training_time)
    logger.info("Outputs saved to %s", output_dir)
    return output_dir
