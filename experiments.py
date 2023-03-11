from typing import List

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
from utils import (
    compute_binomial_ci,
    compute_ogd_wo_grad_constants,
    compute_ogd_w_grad_constants,
)


@timeit
def run_ogd_comparison(
    dim: int,
    horizon: int,
    lipschitz_constant: float,
    diameter: float,
    n_trials: int,
    log_plots: bool,
) -> None:
    delta, eta = compute_ogd_wo_grad_constants(
        dim, horizon, lipschitz_constant, diameter
    )
    eta_ogd = compute_ogd_w_grad_constants(horizon, lipschitz_constant, diameter)

    wo_grad, w_grad = np.zeros((n_trials, horizon)), np.zeros((n_trials, horizon))
    for trial in range(n_trials):
        wo_grad[trial] = OGDWithoutGradient(
            dim=dim, delta=delta, eta=eta
        ).play_full_horizon(horizon=horizon)
        w_grad[trial] = OGDWithGradient(dim=dim, eta=eta_ogd).play_full_horizon(
            horizon=horizon
        )

    wo_grad_mean, wo_grad_std = wo_grad.mean(axis=0), wo_grad.std(axis=0)
    w_grad_mean, w_grad_std = w_grad.mean(axis=0), w_grad.std(axis=0)

    plt.figure(figsize=(10, 10))
    plt.plot(wo_grad_mean, label="OGD without gradient")
    plt.fill_between(
        np.arange(horizon),
        wo_grad_mean - 1.96 * wo_grad_std / np.sqrt(n_trials),
        wo_grad_mean + 1.96 * wo_grad_std / np.sqrt(n_trials),
        alpha=0.5,
        label="95% CI",
    )
    plt.plot(w_grad_mean, label="OGD with gradient")
    plt.fill_between(
        np.arange(horizon),
        w_grad_mean - 1.96 * w_grad_std / np.sqrt(n_trials),
        w_grad_mean + 1.96 * w_grad_std / np.sqrt(n_trials),
        alpha=0.5,
        label="95% CI",
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
    eta_ogd = compute_ogd_w_grad_constants(horizon, lipschitz_constant, diameter)

    wo_grad = np.zeros((n_trials, dims.shape[0]))
    w_grad = np.zeros((n_trials, dims.shape[0]))

    for i, dim in enumerate(dims):
        delta, eta = compute_ogd_wo_grad_constants(
            dim, horizon, lipschitz_constant, diameter
        )
        for trial in range(n_trials):
            wo_grad[trial][i] = OGDWithoutGradient(
                dim=dim, delta=delta, eta=eta
            ).play_full_horizon(horizon=horizon)[-1]
            w_grad[trial][i] = OGDWithGradient(dim=dim, eta=eta_ogd).play_full_horizon(
                horizon=horizon
            )[-1]

    wo_grad_mean, wo_grad_std = wo_grad.mean(axis=0), wo_grad.std(axis=0)
    w_grad_mean, w_grad_std = w_grad.mean(axis=0), w_grad.std(axis=0)

    # noinspection PyArgumentEqualDefault
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.plot(dims, wo_grad_mean, label="OGD without gradient")
    ax1.fill_between(
        dims,
        wo_grad_mean - 1.96 * wo_grad_std / np.sqrt(n_trials),
        wo_grad_mean + 1.96 * wo_grad_std / np.sqrt(n_trials),
        alpha=0.5,
        label="95% CI",
    )
    ax2.plot(dims, w_grad_mean, label="OGD with gradient")
    ax2.fill_between(
        dims,
        w_grad_mean - 1.96 * w_grad_std / np.sqrt(n_trials),
        w_grad_mean + 1.96 * w_grad_std / np.sqrt(n_trials),
        alpha=0.5,
        label="95% CI",
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
    best_arm = np.argmax(arm_means)

    uniform = np.zeros((len(stopping_times), n_trials))
    successive_rejects = np.zeros((len(stopping_times), n_trials))

    timer = checkpoint()
    for tau_i, tau in enumerate(stopping_times):
        for trial in range(n_trials):
            uniform[tau_i][trial] = (
                UniformSamplingFixedBudget(tau, bandit)() == best_arm
            )
            successive_rejects[tau_i][trial] = (
                SuccessiveRejects(tau, bandit)() == best_arm
            )
        timer(f"Time spent on horizon {tau}")

    uniform_means = uniform.mean(axis=1)
    sr_means = successive_rejects.mean(axis=1)

    print(
        f"\nStopping times: {', '.join(str(tau) for tau in stopping_times)}\n"
        f"Success rate of the uniform sampling:   {str(uniform_means)}\n"
        f"Success rate of the successive rejects: {str(sr_means)}"
    )

    plt.figure(figsize=(10, 10))
    plt.errorbar(
        uniform_means,
        stopping_times,
        xerr=compute_binomial_ci(uniform_means, n_trials),
        label="Uniform Sampling",
        fmt="o",
    )
    plt.errorbar(
        sr_means,
        stopping_times,
        xerr=compute_binomial_ci(sr_means, n_trials),
        label="Successive Rejects",
        fmt="o",
    )
    plt.xlabel(r"$\mathbb{P}_\nu[\hat{k} \neq k^*]$")
    plt.ylabel(r"$\tau$")
    plt.legend()
    plt.title("Comparison of Successive Rejects and Uniform sampling")
    plt.show()


@timeit
def run_ucb(
    arm_means: List[float],
    horizon: int,
    n_trials: int,
    plot_bound: bool = True,
) -> None:
    cumulative_rewards = np.zeros((n_trials, horizon))
    bandit = StochasticBandit(arm_means)

    for trial in range(n_trials):
        cumulative_rewards[trial] = UCB(bandit)(horizon)
    mean_regret = (
        max(arm_means) * np.ones(horizon)
    ).cumsum() - cumulative_rewards.mean(axis=0)
    rewards_std = cumulative_rewards.std(axis=0)

    plt.figure(figsize=(10, 10))
    plt.plot(mean_regret, label=r"$\mathbb{E}[R_t]$")
    if plot_bound:
        plt.plot(
            np.sqrt(
                len(arm_means)
                * np.arange(horizon)
                * (np.log(np.arange(horizon) + 1e-3))
            ),
            label=r"$\sqrt{KT\log{T}}$",
        )
    plt.fill_between(
        np.arange(horizon),
        mean_regret - 1.96 * rewards_std / np.sqrt(n_trials),
        mean_regret + 1.96 * rewards_std / np.sqrt(n_trials),
        alpha=0.5,
        label="95% CI",
    )
    plt.title("UCB regret")
    plt.ylabel(r"$\mathbb{E}[R_t]$")
    plt.xlabel("$t$")
    plt.legend()
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
        uniform_time[trial], uniform_error[trial] = UniformSamplingFixedConfidence(
            confidence_level=confidence_level, env=bandit
        )()
        ucb_time[trial], ucb_error[trial] = UCBFixedConfidence(
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
