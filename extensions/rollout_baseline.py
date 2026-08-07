"""
Greedy Rollout Baseline (Kool et al., 2019)

WHY THIS BASELINE IS THE GOLD STANDARD FOR NCO:
Instead of an Exponential Moving Average of past scalar rewards (which is noisy), Kool et al. 
proposed the Greedy Rollout Baseline:

1. Maintain a frozen copy of the best policy model seen so far ('baseline_policy').
2. For each instance i in a training batch, run 'baseline_policy' in GREEDY mode to compute 
   a deterministic baseline reward b_i = R_baseline(i).
3. Compute Advantage A_i = R_sample(i) - R_baseline_greedy(i).
4. At the end of every epoch, evaluate the candidate policy against the baseline policy on a 
   fixed validation set. If the candidate achieves a statistically significant improvement 
   (paired t-test p < 0.05), update baseline_policy = copy(candidate_policy)!

This provides an exact, instance-specific baseline signal that forces the network to 
out-perform its own best deterministic greedy strategy.
"""

import copy
import torch
import numpy as np
from scipy import stats
from typing import List, Tuple
from env.nesting_env import NestingEnv
from model.policy import AttentionPolicy


class GreedyRolloutBaseline:
    def __init__(self, policy: AttentionPolicy, env: NestingEnv):
        """
        Parameters:
        -----------
        policy : AttentionPolicy
            The initial policy network to clone.
        env : NestingEnv
            Environment instance for evaluation.
        """
        self.env = env
        # Deepcopy current policy to serve as frozen baseline model
        self.baseline_policy = copy.deepcopy(policy)
        self.baseline_policy.eval()

    def eval(self, batch_tensor: torch.Tensor, batch_pieces_np: np.ndarray) -> np.ndarray:
        """
        Runs the frozen baseline policy in GREEDY mode to get instance-specific baseline rewards.

        Parameters:
        -----------
        batch_tensor : torch.Tensor of shape (batch_size, N, features)
            PyTorch input tensor for current batch.
        batch_pieces_np : np.ndarray of shape (batch_size, N, 2)
            Raw piece dimensions.

        Returns:
        --------
        baseline_rewards : np.ndarray of shape (batch_size,)
            Greedy utilization scores achieved by baseline policy.
        """
        self.baseline_policy.eval()
        batch_size = len(batch_pieces_np)
        baseline_rewards = []

        with torch.no_grad():
            actions, _, _ = self.baseline_policy(batch_tensor, decode_type="greedy")
            actions_np = actions.cpu().numpy()

        for i in range(batch_size):
            state = self.env.reset(pieces=batch_pieces_np[i])
            for act in actions_np[i]:
                state, reward, done, _ = self.env.step(act)
            baseline_rewards.append(self.env.score())

        return np.array(baseline_rewards, dtype=np.float32)

    def update_baseline(
        self,
        candidate_policy: AttentionPolicy,
        val_instances: List[np.ndarray],
        p_val_threshold: float = 0.05
    ) -> bool:
        """
        Evaluates candidate policy vs. baseline policy on fixed validation set using paired t-test.

        Returns True if baseline policy was updated.
        """
        candidate_policy.eval()
        self.baseline_policy.eval()

        candidate_scores = []
        baseline_scores = []

        with torch.no_grad():
            for inst in val_instances:
                inst_tensor = torch.tensor(inst[np.newaxis, :, :], dtype=torch.float32)

                # Candidate greedy score
                c_act, _, _ = candidate_policy(inst_tensor, decode_type="greedy")
                state = self.env.reset(pieces=inst)
                for act in c_act[0].cpu().numpy():
                    state, reward, done, _ = self.env.step(act)
                candidate_scores.append(self.env.score())

                # Baseline greedy score
                b_act, _, _ = self.baseline_policy(inst_tensor, decode_type="greedy")
                state = self.env.reset(pieces=inst)
                for act in b_act[0].cpu().numpy():
                    state, reward, done, _ = self.env.step(act)
                baseline_scores.append(self.env.score())

        c_scores = np.array(candidate_scores)
        b_scores = np.array(baseline_scores)

        c_mean, b_mean = np.mean(c_scores), np.mean(b_scores)

        # Perform paired t-test to check if candidate is statistically superior
        if c_mean > b_mean:
            t_stat, p_val = stats.ttest_rel(c_scores, b_scores)
            if p_val < p_val_threshold or np.isnan(p_val):
                # Update baseline model to clone of candidate policy!
                self.baseline_policy = copy.deepcopy(candidate_policy)
                self.baseline_policy.eval()
                print(f"\n[*] BASELINE UPDATED! Candidate Mean: {c_mean:.2f}% vs Old Baseline: {b_mean:.2f}% (p-val: {p_val:.4f})")
                return True

        return False
