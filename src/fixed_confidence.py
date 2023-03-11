from abc import ABC, abstractmethod
from typing import Tuple, Literal

import numpy as np

from .discrete_case import StochasticBandit
from .ucb import UCB


class BaseAlgo(ABC):
    def __init__(self, confidence_level: float, env: StochasticBandit):
        self.env = env
        self.n_arms = env.n_arms
        self.confidence_level = confidence_level

    def stopping_rule(
        self,
        empirical_means: np.ndarray,
        n_visits: np.ndarray,
        time: int,
    ) -> bool:
        best_arm = empirical_means.argmax()
        mask = np.ones(self.n_arms, dtype=bool)
        mask[best_arm] = False
        return (
            (empirical_means[best_arm] - empirical_means[mask]) ** 2
            / (1 / n_visits[best_arm] + 1 / n_visits[mask])
        ).min() / 2 > np.log(1 / self.confidence_level) + 3 * np.log(1 + np.log(time))

    @abstractmethod
    def __call__(self) -> Tuple[int, int]:
        pass


class UCBFixedConfidence(BaseAlgo, UCB):
    def __call__(self) -> Tuple[int, int]:
        empirical_means, n_visits, time = self.pull_each_arm_once()

        while not self.stopping_rule(empirical_means, n_visits, time):
            time += 1
            self.play_one_round(empirical_means, n_visits, time)

        return time, int(n_visits.argmax())


class UniformSamplingFixedConfidence(BaseAlgo, UCB):
    def __call__(self) -> Tuple[int, int]:
        empirical_means, n_visits, time = self.pull_each_arm_once()

        while not self.stopping_rule(empirical_means, n_visits, time):
            arm = time % self.n_arms
            empirical_means[arm] = (
                empirical_means[arm] * n_visits[arm] + self.env.observe_gaussian(arm)
            ) / (n_visits[arm] + 1)
            n_visits[arm] += 1
            time += 1

        return time, int(empirical_means.argmax())


class TopTwo(BaseAlgo, UCB):
    def __init__(
        self,
        confidence_level: float,
        env: StochasticBandit,
        leader_selection: Literal["TTUCB", "EB-TC"],
        beta: float = 0.5,
    ):
        super(TopTwo, self).__init__(confidence_level, env)
        self.beta = beta
        self.leader_selection = leader_selection.upper()

        if leader_selection.upper() not in ["TTUCB", "EB-TC"]:
            raise ValueError("Incorrect value for leader selection rule.")

    def __call__(self, max_iter: int = 1e5) -> Tuple[int, int]:
        empirical_means, n_visits, time = self.pull_each_arm_once()

        while not self.stopping_rule(empirical_means, n_visits, time):
            if time >= max_iter:
                print(f"\nMaximum iteration number reached with {self.leader_selection}.")
                break
            time += 1
            # arm selected according to either UCB or empirical means observation
            leader = (
                (empirical_means + np.sqrt(2 * np.log(time) / n_visits)).argmax()
                if self.leader_selection == "TTUCB"
                else empirical_means.argmax()
            )

            mask = np.ones(self.n_arms, dtype=bool)
            mask[leader] = False
            challenger = np.arange(self.n_arms)[mask][
                (
                    (empirical_means[leader] - empirical_means[mask]) ** 2
                    / (1 / n_visits[leader] + 1 / n_visits[mask])
                ).argmin()
            ]

            # sampling the leader with probability beta
            selected_arm = leader if np.random.binomial(1, self.beta) else challenger

            # updating the mean and the number of visits
            empirical_means[selected_arm] = (
                empirical_means[selected_arm] * n_visits[selected_arm]
                + self.env.observe_gaussian(selected_arm)
            ) / (n_visits[selected_arm] + 1)
            n_visits[selected_arm] += 1

        return time, int(empirical_means.argmax())
