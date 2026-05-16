from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training import run_toy_pinn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimal toy PINN experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/toy_pinn.yaml"),
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = run_toy_pinn(args.config)
    print(f"Toy PINN outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
