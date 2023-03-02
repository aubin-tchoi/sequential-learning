import argparse
import json
from collections import defaultdict
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np

from src import (
    OGDWithoutGradient,
    OGDWithGradient,
    StochasticBandit,
    SuccessiveRejects,
    UniformSamplingFixedBudget,
    UCBFixedConfidence,
    UniformSamplingFixedConfidence,
    TopTwo,
    timeit,
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
    parser.add_argument(
        "--log_plots",
        action="store_true",
        help="Plots the regret in log log scale",
    )

    return parser.parse_args()


def compute_constants(
    dim: int, horizon: int, lipschitz_constant: float, diameter: float
) -> Tuple[float, float]:
    delta = horizon ** (-1 / 4) * np.sqrt(dim * diameter / (3 * lipschitz_constant))
    eta = horizon ** (-3 / 4) * np.sqrt(dim / (3 * lipschitz_constant * dim)) * diameter
    return delta, eta


@timeit
def run_ogd_comparison(
    dim: int,
    horizon: int,
    lipschitz_constant: float,
    diameter: float,
    n_trials: int,
    log_plots: bool,
) -> None:
    delta, eta = compute_constants(dim, horizon, lipschitz_constant, diameter)
    wo_grad, w_grad = np.zeros((n_trials, horizon)), np.zeros((n_trials, horizon))
    for trial in range(n_trials):
        wo_grad[trial] = OGDWithoutGradient(
            dim=dim, delta=delta, eta=eta
        ).play_full_horizon(horizon=horizon)
        w_grad[trial] = OGDWithGradient(
            dim=dim, delta=delta, eta=eta
        ).play_full_horizon(horizon=horizon)

    wo_grad_mean, wo_grad_std = wo_grad.mean(axis=0), wo_grad.std(axis=0)
    w_grad_mean, w_grad_std = w_grad.mean(axis=0), w_grad.std(axis=0)

    plt.figure(figsize=(10, 10))
    plt.plot(wo_grad_mean, label="OGD without gradient")
    plt.fill_between(
        np.arange(horizon),
        wo_grad_mean - wo_grad_std,
        wo_grad_mean + wo_grad_std,
        alpha=0.5,
    )
    plt.plot(w_grad_mean, label="OGD with gradient")
    plt.fill_between(
        np.arange(horizon),
        w_grad_mean - w_grad_std,
        w_grad_mean + w_grad_std,
        alpha=0.5,
    )
    if log_plots:
        plt.xscale("log")
        plt.yscale("log")
    plt.legend()
    plt.xlabel("$t$")
    plt.ylabel(r"$\mathbb{E}[R_t]$")
    plt.title("Comparison of OGD and OGD without gradient")
    plt.show()


@timeit
def run_ogd_over_dims(
    dims: np.ndarray,
    horizon: int,
    lipschitz_constant: float,
    diameter: float,
    n_trials: int,
) -> None:
    wo_grad = np.zeros((n_trials, dims.shape[0]))
    w_grad = np.zeros((n_trials, dims.shape[0]))

    for i, dim in enumerate(dims):
        delta, eta = compute_constants(dim, horizon, lipschitz_constant, diameter)
        for trial in range(n_trials):
            wo_grad[trial][i] = OGDWithoutGradient(
                dim=dim, delta=delta, eta=eta
            ).play_full_horizon(horizon=horizon)[-1]
            w_grad[trial][i] = OGDWithGradient(
                dim=dim, delta=delta, eta=eta
            ).play_full_horizon(horizon=horizon)[-1]

    wo_grad_mean, wo_grad_std = wo_grad.mean(axis=0), wo_grad.std(axis=0)
    w_grad_mean, w_grad_std = w_grad.mean(axis=0), w_grad.std(axis=0)

    # noinspection PyArgumentEqualDefault
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.plot(dims, wo_grad_mean, label="OGD without gradient")
    ax1.fill_between(
        dims, wo_grad_mean - wo_grad_std, wo_grad_mean + wo_grad_std, alpha=0.5
    )
    ax2.plot(dims, w_grad_mean, label="OGD with gradient")
    ax2.fill_between(
        dims, w_grad_mean - w_grad_std, w_grad_mean + w_grad_std, alpha=0.5
    )

    ax1.set(xlabel="$d$", ylabel=r"$\mathbb{E}[R_T]$")
    ax1.legend()
    ax2.set(xlabel="$d$", ylabel=r"$\mathbb{E}[R_T]$")
    ax2.legend()
    plt.show()


@timeit
def run_fixed_budget(
    arm_means: List[float], stopping_times: List[int], n_trials: int
) -> None:
    bandit = StochasticBandit(arm_means)
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


@timeit
def run_ucb(arm_means: List[float], horizon: int, n_trials: int) -> None:
    cumulative_rewards = np.zeros((n_trials, horizon))
    bandit = StochasticBandit(arm_means)

    for trial in range(n_trials):
        cumulative_rewards[trial] = UCB(bandit)(horizon)
    mean_regret = (
        max(arm_means) * np.ones(horizon)
    ).cumsum() - cumulative_rewards.mean(axis=0)
    rewards_std = cumulative_rewards.std(axis=0)

    plt.figure(figsize=(10, 10))
    plt.plot(mean_regret)
    plt.fill_between(
        np.arange(horizon),
        mean_regret - rewards_std,
        mean_regret + rewards_std,
        alpha=0.5,
    )
    plt.title("UCB regret")
    plt.ylabel(r"$\mathbb{E}[R_t]$")
    plt.xlabel("$t$")
    plt.show()


@timeit
def run_fixed_confidence(
    confidence_level: float, arm_means: List[float], n_trials: int
) -> None:
    ucb_time, uniform_time = np.zeros(n_trials), np.zeros(n_trials)
    ucb_error, uniform_error = np.zeros(n_trials), np.zeros(n_trials)
    ttucb_time, ebtc_time = np.zeros(n_trials), np.zeros(n_trials)
    ttucb_error, ebtc_error = np.zeros(n_trials), np.zeros(n_trials)
    bandit = StochasticBandit(arm_means)

    for trial in range(n_trials):
        ucb_time[trial], ucb_error[trial] = UCBFixedConfidence(
            confidence_level=confidence_level, env=bandit
        )()
        uniform_time[trial], uniform_error[trial] = UniformSamplingFixedConfidence(
            confidence_level=confidence_level, env=bandit
        )()
        ttucb_time[trial], ttucb_error[trial] = TopTwo(
            confidence_level=confidence_level, env=bandit, leader_selection="TTUCB"
        )()
        ebtc_time[trial], ebtc_error[trial] = TopTwo(
            confidence_level=confidence_level, env=bandit, leader_selection="EB-TC"
        )()

    best_arm = np.argmax(arm_means)
    ucb_error = ucb_error != best_arm
    uniform_error = uniform_error != best_arm
    print(
        f"Mean error rate of uniform sampling: {ucb_error.mean() * 100:.2f}%, std: {ucb_error.std():2f}"
    )
    print(
        f"Mean error rate of UCB:              {uniform_error.mean() * 100:.2f}%, std: {uniform_error.std():2f}"
    )
    ttucb_error = ttucb_error != best_arm
    ebtc_error = ebtc_error != best_arm
    print(
        f"Mean error rate of TTUCB:            {ttucb_error.mean() * 100:.2f}%, std: {ttucb_error.std():2f}"
    )
    print(
        f"Mean error rate of EB-TC:            {ebtc_error.mean() * 100:.2f}%, std: {ebtc_error.std():2f}"
    )
    plt.figure()
    plt.boxplot([ucb_time, uniform_time], labels=["UCB", "Uniform sampling"])
    plt.ylabel("Stopping time")
    plt.legend()
    plt.title("Comparison of two fixed confidence algorithms")
    plt.show()

    plt.figure()
    plt.boxplot(
        [ucb_time, uniform_time, ttucb_time, ebtc_time],
        labels=["UCB", "Uniform sampling", "TTUCB", "EB-TC"],
    )
    plt.ylabel("Stopping time")
    plt.legend()
    plt.title("Comparison of four fixed confidence algorithms")
    plt.show()


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
