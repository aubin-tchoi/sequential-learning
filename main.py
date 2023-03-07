import numpy as np

from experiments import (
    run_ogd_comparison,
    run_ogd_over_dims,
    run_fixed_budget,
    run_ucb,
    run_fixed_confidence,
)
from utils import parse_args

if __name__ == "__main__":
    args = parse_args()

    if not args.skip_ogd:
        run_ogd_comparison(
            dim=2,
            horizon=1000,
            lipschitz_constant=5,
            diameter=2,
            n_trials=500,
            log_plots=args.log_plots,
        )
        run_ogd_over_dims(
            dims=np.arange(10) + 1,
            horizon=1000,
            lipschitz_constant=5,
            diameter=2,
            n_trials=50,
        )

    if not args.skip_fixed_budget:
        means = [0.4 for _ in range(19)]
        means.insert(0, 0.5)
        run_fixed_budget(arm_means=means, stopping_times=[100, 200, 500], n_trials=5000)

    if not args.skip_ucb:
        means = [0.5, 0.4, 0.4]
        for _ in range(7):
            means.append(0.3)
        assert len(means) == 10, "Incorrect number of arms"
        run_ucb(arm_means=means, horizon=10000, n_trials=100)

    if not args.skip_fixed_confidence:
        means = [0.5, 0.4, 0.4]
        for _ in range(7):
            means.append(0.3)
        assert len(means) == 10, "Incorrect number of arms"
        run_fixed_confidence(confidence_level=0.01, arm_means=means, n_trials=1)
