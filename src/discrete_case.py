"""
Algorithms for stochastic best arm identification in a bandit setting.
We assume the decision set to be finite and the rewards to only depend on the arm (independent of the time).
"""
from typing import List

import numpy as np


class StochasticBandit:
    def __init__(self, means: List[float], sigma: float = 0.1):
        self.n_arms = len(means)
        self.arm_means = means

    def observe(self, arm: int) -> float:
        return np.random.binomial(1, self.arm_means[arm])


class UniformSampling:
    def __init__(self, stopping_time: int, env: StochasticBandit):
        self.env = env
        self.n_rounds = stopping_time // env.n_arms

    def play(self) -> int:
        rewards = np.zeros(self.env.n_arms)
        for arm in range(self.env.n_arms):
            for t in range(self.n_rounds):
                rewards[arm] += self.env.observe(arm)

        return int(rewards.argmax())


class SuccessiveRejects:
    def __init__(self, stopping_time: int, env: StochasticBandit):
        self.env = env
        self.stopping_time = stopping_time
        self.n_arms = self.env.n_arms
        self.logK = 0.5 + sum(1 / k for k in range(2, env.n_arms + 1))

    def play(self, verbose: bool = False) -> int:
        remaining_arms = list(range(self.n_arms))
        n = 0
        for j in range(1, self.n_arms):
            if verbose:
                print(
                    f"Remaining arms: {', '.join(f'{str(arm):>5}' for arm in remaining_arms)}"
                )

            next_n = int(
                np.ceil(
                    1
                    / self.logK
                    * (self.stopping_time - self.n_arms)
                    / (self.n_arms + 1 - j)
                )
            )
            rewards = np.zeros(len(remaining_arms))
            for i, arm in enumerate(remaining_arms):
                for t in range(next_n - n):
                    rewards[i] += self.env.observe(arm)

            if verbose:
                print(
                    f"Arm scores:     {', '.join(f'{str(round(r, 2)):>5}' for r in rewards)}"
                )

            rejected_arm = np.random.choice(np.where(rewards == rewards.min())[0])
            remaining_arms.pop(rejected_arm)
            n = next_n

            if verbose:
                print(f"Rejecting arm {remaining_arms[rejected_arm]}.")

        return remaining_arms.pop()
