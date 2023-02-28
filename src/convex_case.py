"""
Algorithms for online zero-order convex optimization.
We assume the decision set to be convex, the loss functions to be convex and G-Lipschitz.
"""
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class BaseOGD(ABC):
    def __init__(self, dim: int, delta: float, eta: float):
        self.dim = dim
        self.theta_star = 1 / (np.arange(dim) + 1) / 2
        self.delta, self.eta = delta, eta

    def sample_direction(self) -> np.ndarray:
        """
        Samples a random direction (vector on the unit sphere).
        """
        direction = np.random.randn(self.dim)

        return direction / np.linalg.norm(direction)

    def sample_loss(self, sigma: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample elements that constitute a loss.
        """
        x_t = np.random.randn(self.dim)
        epsilon_t = np.random.randn(1) * sigma
        y_t = self.theta_star @ x_t + epsilon_t

        return x_t, y_t

    @staticmethod
    def compute_loss(theta: np.ndarray, x_t: np.ndarray, y_t: np.ndarray) -> float:
        """
        Computes the loss incurred by theta.
        """
        return float((theta @ x_t - y_t) ** 2)

    @staticmethod
    def euclidian_projection(theta: np.ndarray, radius: float) -> np.ndarray:
        """
        Euclidian projection onto a sphere of given radius.
        """
        theta_norm = np.linalg.norm(theta)
        if theta_norm > radius:
            return theta / theta_norm * radius
        return theta

    @abstractmethod
    def update_rule(
        self,
        theta_hat: np.ndarray,
        direction: np.ndarray,
        x_t: np.ndarray,
        y_t: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Updates theta_hat according to the observations of x_t, y_t and to the sampled direction.

        Returns:
            theta_hat (np.ndarray): next value of theta_hat.
            regret (float): difference between the observed loss and the loss of theta_star.
        """
        pass

    def play_one_episode(self, theta_hat: np.ndarray) -> Tuple[np.ndarray, float]:
        direction = self.sample_direction()
        x_t, y_t = self.sample_loss()

        return self.update_rule(theta_hat, direction, x_t, y_t)

    def play_full_horizon(self, horizon: int) -> np.ndarray:
        theta_hat, regret = np.zeros(self.dim), np.zeros(horizon)
        for t in range(horizon):
            theta_hat, regret[t] = self.play_one_episode(theta_hat)

        return regret.cumsum()


class OGDWithoutGradient(BaseOGD):
    def update_rule(
        self,
        theta_hat: np.ndarray,
        direction: np.ndarray,
        x_t: np.ndarray,
        y_t: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Updates theta_hat according to a sample of loss.

        Returns:
            theta_hat: new position for theta_hat
            regret: loss observed discounted by the loss that would have been observed by theta star.
        """
        theta_t = theta_hat + self.delta * direction
        loss = self.compute_loss(theta_t, x_t, y_t)

        return (
            self.euclidian_projection(
                theta_hat - self.dim * self.eta / self.delta * loss * direction,
                1 - self.delta
            ),
            loss - self.compute_loss(self.theta_star, x_t, y_t),
        )


class OGDWithGradient(BaseOGD):
    def update_rule(
        self,
        theta_hat: np.ndarray,
        direction: np.ndarray,
        x_t: np.ndarray,
        y_t: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Updates theta_hat according to a sample of loss.

        Returns:
            theta_hat: new position for theta_hat
            regret: loss observed discounted by the loss that would have been observed by theta star.
        """
        loss = self.compute_loss(theta_hat, x_t, y_t)
        return (
            self.euclidian_projection(
                theta_hat - self.eta * 2 * x_t * (theta_hat @ x_t - y_t),
                1
            ),
            loss - self.compute_loss(self.theta_star, x_t, y_t),
        )
