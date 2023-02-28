import json
from collections import defaultdict

import matplotlib.pyplot as plt

from src import (
    OGDWithoutGradient,
    OGDWithGradient,
    StochasticBandit,
    SuccessiveRejects,
    UniformSampling,
    checkpoint,
)

# TODO: add argparse

if __name__ == "__main__":
    run_ogd = False
    run_bandit = True

    if run_ogd:
        d, T = 2, 1000
        delta = 2 * T / d
        eta = 4 * T / d**2

        fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 12))

        ogd_wo_grad = OGDWithoutGradient(dim=d, delta=delta, eta=eta)
        ogd_wo_grad_regret = ogd_wo_grad.play_full_horizon(horizon=T)
        ax1.plot(ogd_wo_grad_regret)

        ogd_w_grad = OGDWithGradient(dim=d, delta=delta, eta=eta)
        ogd_w_grad_regret = ogd_w_grad.play_full_horizon(horizon=T)
        ax2.plot(ogd_w_grad_regret)

        plt.show()

    if run_bandit:
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
                uniform[tau][UniformSampling(tau, bandit).play()] += 1
                successive_rejects[tau][SuccessiveRejects(tau, bandit).play()] += 1
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
