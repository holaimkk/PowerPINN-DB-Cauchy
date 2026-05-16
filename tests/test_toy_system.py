import torch

from src.systems import ExponentialDecayODE


def test_exact_solution_initial_value() -> None:
    system = ExponentialDecayODE(t_min=0.0, t_max=1.0, initial_t=0.0, initial_u=1.0)
    t = torch.tensor([[0.0]])
    assert torch.allclose(system.exact_solution(t), torch.tensor([[1.0]]))
