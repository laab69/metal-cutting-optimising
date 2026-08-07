"""
Exponential Moving Average Baseline (`moving_average.py`)

WHY THIS BASELINE EXISTS:
In REINFORCE (Williams, 1992), policy gradient variance is high because all rewards are 
positive scalar values (e.g., 55% to 75% utilization).

Subtracting a baseline reward b transforms raw rewards R into Advantage values (R - b):
- Positive Advantage (+A): Trajectory beat expectations -> increase action probability.
- Negative Advantage (-A): Trajectory fell below expectations -> decrease action probability.

Using an Exponential Moving Average (EMA) baseline introduces ZERO bias into the gradient 
estimator while dramatically speeding up training convergence (Bello et al. 2016).
"""

import numpy as np


class MovingAverageBaseline:
    def __init__(self, beta: float = 0.95):
        """
        Parameters:
        -----------
        beta : float
            Decay factor for exponential smoothing (default: 0.95).
        """
        self.beta = beta
        self.value: float = None

    def update(self, batch_rewards: np.ndarray) -> float:
        """
        Updates the moving average baseline using the current batch's mean reward.

        Parameters:
        -----------
        batch_rewards : np.ndarray
            Array of scalar rewards (utilization %) from current training batch.

        Returns:
        --------
        current_baseline_value : float
        """
        batch_mean = float(np.mean(batch_rewards))

        if self.value is None:
            # Initialize baseline to first batch mean
            self.value = batch_mean
        else:
            # Exponential moving average update rule
            self.value = self.beta * self.value + (1.0 - self.beta) * batch_mean

        return self.value

    def get_advantage(self, batch_rewards: np.ndarray) -> np.ndarray:
        """
        Computes advantage A_i = R_i - b for each trajectory in the batch.
        """
        if self.value is None:
            return batch_rewards - np.mean(batch_rewards)
        return batch_rewards - self.value
