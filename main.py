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

# TODO: add argparse
# TODO: add legends to plots

if __name__ == "__main__":
    run_ogd = False
    run_fixed_budget = False
    run_fixed_confidence = True

    if run_ogd:
        d, T = 2, 1000
        delta = 2 * T / d
        eta = 4 * T / d**2

        fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 12))

        ogd_wo_grad_regret = OGDWithoutGradient(
            dim=d, delta=delta, eta=eta
        ).play_full_horizon(horizon=T)
        ax1.plot(ogd_wo_grad_regret)

        ogd_w_grad_regret = OGDWithGradient(
            dim=d, delta=delta, eta=eta
        ).play_full_horizon(horizon=T)
        ax2.plot(ogd_w_grad_regret)

        plt.show()

    if run_fixed_budget:
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

    if run_fixed_confidence:
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
