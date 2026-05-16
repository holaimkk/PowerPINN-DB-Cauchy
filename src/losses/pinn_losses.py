from __future__ import annotations

import torch

from src.systems.simple_ode import ExponentialDecayODE


def compute_toy_pinn_losses(
    model: torch.nn.Module,
    system: ExponentialDecayODE,
    t_collocation: torch.Tensor,
    physics_weight: float,
    initial_condition_weight: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    residual = system.residual(model, t_collocation)
    physics_loss = torch.mean(residual**2)

    t0, u0 = system.initial_point(device)
    initial_loss = torch.mean((model(t0) - u0) ** 2)

    total_loss = physics_weight * physics_loss + initial_condition_weight * initial_loss
    return {
        "total_loss": total_loss,
        "physics_loss": physics_loss,
        "initial_condition_loss": initial_loss,
    }
