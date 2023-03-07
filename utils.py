import argparse
from typing import Tuple, Union

import numpy as np


def compute_binomial_ci(mean: Union[float, np.ndarray], n_trials: int) -> Union[float, np.ndarray]:
    return 1.96 * np.sqrt(mean * (1 - mean) / n_trials)


def compute_ogd_wo_grad_constants(
    dim: int, horizon: int, lipschitz_constant: float, diameter: float
) -> Tuple[float, float]:
    delta = horizon ** (-1 / 4) * np.sqrt(dim * diameter / (3 * lipschitz_constant))
    eta = horizon ** (-3 / 4) * np.sqrt(dim / (3 * lipschitz_constant * dim)) * diameter

    return delta, eta


def compute_ogd_w_grad_constants(
    horizon: int, lipschitz_constant: float, diameter: float
) -> float:
    eta = diameter / (lipschitz_constant * np.sqrt(horizon))

    return eta


def parse_args() -> argparse.Namespace:
    """
    Parses the command line arguments and produces the help message.
    """
    parser = argparse.ArgumentParser(description="Surface reconstruction")

    parser.add_argument(
        "--skip_ogd",
        action="store_true",
        help="Skips the execution of the two OGD algorithms",
    )
    parser.add_argument(
        "--skip_fixed_budget",
        action="store_true",
        help="Skips the execution of the fixed budget algorithms",
    )
    parser.add_argument(
        "--skip_ucb",
        action="store_true",
        help="Skips the execution of the UCB algorithm",
    )
    parser.add_argument(
        "--skip_fixed_confidence",
        action="store_true",
        help="Skips the execution of the two fixed confidence algorithms",
    )
    parser.add_argument(
        "--log_plots",
        action="store_true",
        help="Plots the regret in log log scale",
    )

    return parser.parse_args()
