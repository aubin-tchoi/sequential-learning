import numpy as np

from .discrete_case import StochasticBandit


class UniformSamplingFixedBudget:
    def __init__(self, stopping_time: int, env: StochasticBandit):
        self.env = env
        self.n_rounds = stopping_time // env.n_arms

    def __call__(self) -> int:
        rewards = np.zeros(self.env.n_arms)
        for arm in range(self.env.n_arms):
            for t in range(self.n_rounds):
                rewards[arm] += self.env.observe_bernoulli(arm)

        return int(rewards.argmax())


class SuccessiveRejects:
    def __init__(self, stopping_time: int, env: StochasticBandit):
        self.env = env
        self.stopping_time = stopping_time
        self.n_arms = self.env.n_arms
        self.logK = 0.5 + sum(1 / k for k in range(2, env.n_arms + 1))

    def __call__(self, verbose: bool = False) -> int:
        if verbose:
            print(" --- Beginning of a simulation ---\n")

        remaining_arms = list(range(self.n_arms))
        rewards = np.zeros(self.n_arms)
        n = 0

        for j in range(1, self.n_arms):
            next_n = int(
                np.ceil(
                    1
                    / self.logK
                    * (self.stopping_time - self.n_arms)
                    / (self.n_arms + 1 - j)
                )
            )

            if verbose:
                print(f"Number of rounds per arm: {next_n - n}")
                print(
                    f"Remaining arms: {', '.join(f'{str(arm):>5}' for arm in remaining_arms)}"
                )

            for arm in remaining_arms:
                for t in range(next_n - n):
                    rewards[arm] += self.env.observe_bernoulli(arm)

            if verbose:
                print(
                    f"Arm scores:     "
                    f"{', '.join(f'{str(round(r, 2)):>5}' for arm, r in enumerate(rewards) if arm in remaining_arms)}"
                )

            argmin = np.where(rewards[remaining_arms] == rewards[remaining_arms].min())[
                0
            ]
            if verbose:
                print(
                    f"Arms with the lowest score: {', '.join(str(remaining_arms[arm]) for arm in argmin)}"
                )
            # choosing an arm randomly among the ones with the lowest score
            rejected_arm = np.random.choice(argmin)
            if verbose:
                print(f"Rejecting arm {remaining_arms[rejected_arm]}.\n")

            remaining_arms.pop(rejected_arm)
            n = next_n

        assert len(remaining_arms) == 1, (
            "There is more than 1 arm left"
            if len(remaining_arms) > 1
            else "There is no arm left"
        )
        if verbose:
            print(f"Selecting arm {remaining_arms[0]}.\n")

        return remaining_arms.pop()
