import argparse

import numpy as np
import seaborn as sns

from scripts.runners import (
    run_fixed_budget,
    run_fixed_confidence,
    run_ogd_comparison,
    run_ogd_over_dims,
    run_ucb,
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
    parser.add_argument(
        "--log_plots",
        action="store_true",
        help="Plots the regret in log log scale",
    )

    return parser.parse_args()


def main(args: argparse.Namespace = parse_args()) -> None:
    sns.set_theme()

    if not args.skip_ogd:
        print("\n ---------- OGD ----------")
        run_ogd_comparison(
            dim=2,
            horizon=1000,
            lipschitz_constant=5,
            diameter=2,
            n_trials=100,
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
        print("\n ------ Fixed budget ------")
        means = [0.4 for _ in range(19)]
        means.insert(0, 0.5)
        run_fixed_budget(arm_means=means, stopping_times=[100, 200, 500], n_trials=5000)

    if not args.skip_ucb:
        print("\n ---------- UCB ----------")
        means = [0.5, 0.4, 0.4]
        for _ in range(7):
            means.append(0.3)
        assert len(means) == 10, "Incorrect number of arms"
        run_ucb(arm_means=means, horizon=10000, n_trials=100)

    if not args.skip_fixed_confidence:
        print("\n ---- Fixed confidence ----")
        means = [0.5, 0.4, 0.4]
        for _ in range(7):
            means.append(0.3)
        assert len(means) == 10, "Incorrect number of arms"
        run_fixed_confidence(confidence_level=0.01, arm_means=means, n_trials=200)


if __name__ == "__main__":
    main()
