"""
Scaled REINFORCE Trainer with Greedy Rollout Baseline (`scaled_trainer.py`)

WHY THIS SCRIPT EXISTS:
Implements Kool et al. (2019)'s full training setup:
1. Scaled instance size: N = 20 pieces (where heuristics begin to degrade).
2. Kool et al. Greedy Rollout Baseline: Uses a frozen snapshot of the best model to generate 
   instance-specific baseline rewards b_i = R_greedy(i).
3. Paired t-test Baseline Updates: Updates baseline model only when candidate policy achieves 
   statistically significant performance gains (p < 0.05).
"""

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from env.generator import generate_instance
from env.nesting_env import NestingEnv
from baseline.largest_first import run_largest_first_heuristic
from model.policy import AttentionPolicy
from extensions.rollout_baseline import GreedyRolloutBaseline


def train_scaled_policy(
    num_steps: int = 600,
    batch_size: int = 64,
    num_pieces: int = 20,
    lr: float = 1e-4,
    sheet_width: float = 100.0,
    sheet_height: float = 100.0,
    seed: int = 2026,
    checkpoint_path: str = "model/scaled_policy.pt",
    plot_path: str = "scaled_training_curve.png"
):
    print("=" * 70)
    print(f"  SCALED REINFORCE TRAINING (N = {num_pieces} PIECES, ROLLOUT BASELINE)")
    print("=" * 70)

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = NestingEnv(sheet_width=sheet_width, sheet_height=sheet_height, num_pieces=num_pieces)

    policy = AttentionPolicy(
        input_dim=2,
        d_model=128,
        num_heads=8,
        num_layers=2,
        sheet_width=sheet_width,
        sheet_height=sheet_height
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    # Instantiate Kool et al. Greedy Rollout Baseline
    rollout_baseline = GreedyRolloutBaseline(policy=policy, env=env)

    # Generate fixed validation set of 100 instances for rollout baseline comparison
    val_instances = [
        generate_instance(num_pieces=num_pieces, sheet_width=sheet_width, sheet_height=sheet_height, seed=8888 + i)
        for i in range(100)
    ]

    # Pre-calculate Phase B Heuristic Floor on N=20 validation set
    heuristic_val_scores = [run_largest_first_heuristic(env, inst)[0] for inst in val_instances]
    heuristic_floor = float(np.mean(heuristic_val_scores))

    print(f"Training Parameters:")
    print(f"  Pieces Per Instance (N)  : {num_pieces}")
    print(f"  Batch Size               : {batch_size}")
    print(f"  Total Steps              : {num_steps}")
    print(f"  Largest-First Heuristic  : {heuristic_floor:.2f}% (Target to beat at N={num_pieces})")
    print("-" * 70)

    history_steps = []
    history_sample_rewards = []
    history_baseline_rewards = []
    history_greedy_val = []

    t_start = time.time()

    for step in range(1, num_steps + 1):
        policy.train()

        # Step 1: Generate batch of 64 random N=20 instances
        batch_pieces_list = [
            generate_instance(num_pieces=num_pieces, sheet_width=sheet_width, sheet_height=sheet_height)
            for _ in range(batch_size)
        ]
        batch_pieces_np = np.stack(batch_pieces_list, axis=0)  # Shape: (64, 20, 2)
        batch_tensor = torch.tensor(batch_pieces_np, dtype=torch.float32)

        # Step 2: Policy sample rollout
        actions_sample, log_probs_sum, _ = policy(batch_tensor, decode_type="sample")
        actions_np = actions_sample.detach().cpu().numpy()

        # Step 3: Compute sample rewards
        sample_rewards = []
        for i in range(batch_size):
            state = env.reset(pieces=batch_pieces_np[i])
            for act in actions_np[i]:
                state, reward, done, _ = env.step(act)
            sample_rewards.append(env.score())

        sample_rewards_np = np.array(sample_rewards, dtype=np.float32)

        # Step 4: Compute Greedy Rollout Baseline rewards b_i = R_greedy(i)
        baseline_rewards_np = rollout_baseline.eval(batch_tensor, batch_pieces_np)

        # Step 5: Advantage A_i = R_sample(i) - R_baseline_greedy(i)
        advantages_np = sample_rewards_np - baseline_rewards_np
        adv_tensor = torch.tensor(advantages_np, dtype=torch.float32)

        # REINFORCE loss
        loss = - torch.mean(log_probs_sum * adv_tensor)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        history_steps.append(step)
        history_sample_rewards.append(float(np.mean(sample_rewards_np)))
        history_baseline_rewards.append(float(np.mean(baseline_rewards_np)))

        # Step 6: Periodic Baseline Update Evaluation (every 50 steps)
        if step % 50 == 0 or step == num_steps:
            # Check paired t-test against baseline model on validation set
            rollout_baseline.update_baseline(policy, val_instances, p_val_threshold=0.05)

            # Evaluate candidate greedy score
            policy.eval()
            val_scores = []
            with torch.no_grad():
                for inst in val_instances:
                    inst_t = torch.tensor(inst[np.newaxis, :, :], dtype=torch.float32)
                    c_act, _, _ = policy(inst_t, decode_type="greedy")
                    state = env.reset(pieces=inst)
                    for act in c_act[0].cpu().numpy():
                        state, reward, done, _ = env.step(act)
                    val_scores.append(env.score())

            mean_greedy_val = float(np.mean(val_scores))
            history_greedy_val.append((step, mean_greedy_val))

            print(f"Step {step:03d}/{num_steps} | Sample Util: {np.mean(sample_rewards_np):.2f}% | Rollout Baseline: {np.mean(baseline_rewards_np):.2f}% | Candidate Greedy Val: {mean_greedy_val:.2f}% (vs Heuristic: {heuristic_floor:.2f}%)")

    elapsed_t = time.time() - t_start
    final_greedy = history_greedy_val[-1][1]

    print("-" * 70)
    print("SCALED TRAINING COMPLETE!")
    print(f"  Elapsed Wall-Clock Time  : {elapsed_t:.2f} seconds")
    print(f"  Largest-First Heuristic  : {heuristic_floor:.2f}%")
    print(f"  Trained Attention Policy : {final_greedy:.2f}%")
    print(f"  Advantage Gain Over Rule : +{final_greedy - heuristic_floor:.2f}% percentage points")
    print("-" * 70)

    # Save Checkpoint
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({'model_state_dict': policy.state_dict(), 'greedy_val': final_greedy}, checkpoint_path)
    print(f"Saved scaled policy checkpoint to '{os.path.abspath(checkpoint_path)}'")

    # Save Learning Curve Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history_steps, history_sample_rewards, label="Sampled Batch Utilization (Training)", color='#1f77b4', alpha=0.3)
    ax.plot(history_steps, history_baseline_rewards, label="Kool et al. Greedy Rollout Baseline", color='#2ca02c', linewidth=2.0)

    v_steps, v_scores = zip(*history_greedy_val)
    ax.plot(v_steps, v_scores, label="Trained Policy Greedy Validation", color='#d62728', linewidth=2.5, marker='o')
    ax.axhline(heuristic_floor, color='orange', linestyle='--', linewidth=2.0, label=f"Largest-First Heuristic Floor ({heuristic_floor:.2f}%)")

    ax.set_xlabel("Training Step", fontweight='bold', fontsize=11)
    ax.set_ylabel("Utilization (%)", fontweight='bold', fontsize=11)
    ax.set_title(f"Scaled NCO Training (N = {num_pieces} Pieces) with Kool et al. Greedy Rollout Baseline", fontweight='bold', fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Scaled learning curve saved to '{os.path.abspath(plot_path)}'\n")
    plt.close()

    return final_greedy

if __name__ == "__main__":
    train_scaled_policy()
