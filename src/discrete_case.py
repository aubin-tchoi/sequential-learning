"""
Algorithms for stochastic best arm identification in a bandit setting.
We assume the decision set to be finite and the rewards to only depend on the arm (independent of the time).
"""
from typing import List

import numpy as np


class StochasticBandit:
    def __init__(self, means: List[float]):
        self.n_arms = len(means)
        self.arm_means = means

    def observe_bernoulli(self, arm: int) -> float:
        return np.random.binomial(1, self.arm_means[arm])

    def observe_gaussian(self, arm: int, sigma: float = 1) -> float:
        return np.random.randn(1) * sigma + self.arm_means[arm]
