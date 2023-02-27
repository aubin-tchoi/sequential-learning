import json
from collections import defaultdict

import matplotlib.pyplot as plt

from src import (
    OGDWithoutGradient,
    OGDWithGradient,
    StochasticBandit,
    SuccessiveRejects,
    UniformSampling,
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
        n_trials = 1000

        uniform = [defaultdict(int) for _ in range(len(stopping_times))]
        successive_rejects = [defaultdict(int) for _ in range(len(stopping_times))]

        for i, stopping_time in enumerate(stopping_times):
            for _ in range(n_trials):
                uniform[i][UniformSampling(stopping_time, bandit).play()] += 1
                successive_rejects[i][
                    SuccessiveRejects(stopping_time, bandit).play()
                ] += 1

        print("Uniform sampling")
        for uni in uniform:
            print(json.dumps(dict(uni), indent=4))

        print("\nSuccessive rejects")
        for suc in successive_rejects:
            print(json.dumps(dict(suc), indent=4))
