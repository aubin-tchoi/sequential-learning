import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from src import (
    OGDWithoutGradient,
    OGDWithGradient,
    StochasticBandit,
    SuccessiveRejects,
    UniformSamplingFixedBudget,
    checkpoint,
    UCB,
)


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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.skip_ogd:
        dim, horizon, lipschitz_constant, diameter = 2, 1000, 5, 2
        delta = horizon ** (-1 / 4) * np.sqrt(dim * diameter / (3 * lipschitz_constant))
        eta = (
            horizon ** (-3 / 4)
            * np.sqrt(dim / (3 * lipschitz_constant * dim))
            * diameter
        )

        plt.figure(figsize=(12, 12))

        ogd_wo_grad_regret = OGDWithoutGradient(
            dim=dim, delta=delta, eta=eta
        ).play_full_horizon(horizon=horizon)
        plt.plot(ogd_wo_grad_regret, label="OGD without gradient")

        ogd_w_grad_regret = OGDWithGradient(
            dim=dim, delta=delta, eta=eta
        ).play_full_horizon(horizon=horizon)
        plt.plot(ogd_w_grad_regret, label="OGD with gradient")

        plt.legend()
        plt.show()

    if not args.skip_fixed_budget:
        means = [0.4 for _ in range(19)]
        means.insert(0, 0.5)

        bandit = StochasticBandit(means)
        stopping_times = [100, 200, 500]
        n_trials = 5000

        uniform = defaultdict(lambda: defaultdict(int))
        successive_rejects = defaultdict(lambda: defaultdict(int))

        timer = checkpoint()
        for tau in stopping_times:
            for _ in range(n_trials):
                uniform[tau][UniformSamplingFixedBudget(tau, bandit)()] += 1
                successive_rejects[tau][SuccessiveRejects(tau, bandit)()] += 1
            timer(f"Time spent on horizon {tau}")

        print(" --- Uniform sampling ---\n")
        for tau, uni in uniform.items():
            print(f"T = {tau}", json.dumps(dict(uni), indent=4))

        print("\n\n --- Successive rejects ---\n")
        for tau, suc in successive_rejects.items():
            print(f"T = {tau}", json.dumps(dict(suc), indent=4))

        print(
            f"\nSuccess rate of the uniform sampling:   "
            f"{', '.join(f'T = {tau}: {uniform[tau][0] / n_trials * 100:>5.2f}%' for tau in stopping_times)}"
        )
        print(
            f"Success rate of the successive rejects: "
            f"{', '.join(f'T = {tau}: {successive_rejects[tau][0] / n_trials * 100:>5.2f}%' for tau in stopping_times)}"
        )

    if not args.skip_ucb:
        means = [0.5, 0.4, 0.4]
        for _ in range(7):
            means.append(0.3)
        assert len(means) == 10, "Incorrect number of arms"

        n_trials = 100
        horizon = 1000

        regret = np.zeros(horizon)
        bandit = StochasticBandit(means)

        for _ in range(n_trials):
            regret += UCB(bandit)(horizon)
        regret /= n_trials
        regret -= (0.3 * np.ones(horizon)).cumsum()

        plt.plot(regret)
        plt.show()
