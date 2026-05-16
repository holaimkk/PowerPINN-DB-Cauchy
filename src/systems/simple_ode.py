from __future__ import annotations

import torch


class ExponentialDecayODE:
    """Toy ODE: u'(t) = -u(t), u(0) = 1."""

    def __init__(self, t_min: float, t_max: float, initial_t: float, initial_u: float) -> None:
        self.t_min = t_min
        self.t_max = t_max
        self.initial_t = initial_t
        self.initial_u = initial_u

    def sample_collocation(self, n_points: int, device: torch.device) -> torch.Tensor:
        t = torch.linspace(self.t_min, self.t_max, n_points, device=device).reshape(-1, 1)
        return t.requires_grad_(True)

    def eval_grid(self, n_points: int, device: torch.device) -> torch.Tensor:
        return torch.linspace(self.t_min, self.t_max, n_points, device=device).reshape(-1, 1)

    def initial_point(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        t0 = torch.tensor([[self.initial_t]], dtype=torch.float32, device=device)
        u0 = torch.tensor([[self.initial_u]], dtype=torch.float32, device=device)
        return t0, u0

    def exact_solution(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(-t)

    def residual(self, model: torch.nn.Module, t: torch.Tensor) -> torch.Tensor:
        u = model(t)
        du_dt = torch.autograd.grad(
            outputs=u,
            inputs=t,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]
        return du_dt + u
