from typing import Tuple

import numpy as np

from .discrete_case import StochasticBandit


class UCB:
    def __init__(self, env: StochasticBandit):
        self.env = env
        self.n_arms = env.n_arms

    def pull_each_arm_once(self) -> Tuple[np.ndarray, np.ndarray, int]:
        empirical_means = np.zeros(self.n_arms)

        # pulling each arm once
        for arm in range(self.n_arms):
            empirical_means[arm] += self.env.observe_gaussian(arm)

        return empirical_means, np.ones(self.n_arms), self.n_arms

    def play_one_round(
        self, empirical_means: np.ndarray, n_visits: np.ndarray, time: int
    ) -> float:
        # arm selection according to UCB
        selected_arm = (empirical_means + np.sqrt(2 * np.log(time) / n_visits)).argmax()

        reward = self.env.observe_gaussian(selected_arm)
        # updating the mean and the number of visits
        empirical_means[selected_arm] = (
            empirical_means[selected_arm] * n_visits[selected_arm] + reward
        ) / (n_visits[selected_arm] + 1)
        n_visits[selected_arm] += 1

        return reward

    def __call__(self, horizon: int) -> np.ndarray:
        empirical_means, n_visits, time = self.pull_each_arm_once()
        rewards = np.zeros(horizon)
        rewards[: self.n_arms] = empirical_means
        while time < horizon:
            time += 1
            rewards[time - 1] = self.play_one_round(empirical_means, n_visits, time)

        return rewards.cumsum()
