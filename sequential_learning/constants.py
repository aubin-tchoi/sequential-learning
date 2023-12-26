from typing import Tuple, Union

import numpy as np


def compute_binomial_ci(
    mean: Union[float, np.ndarray], n_trials: int
) -> Union[float, np.ndarray]:
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
