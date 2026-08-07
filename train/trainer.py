"""
REINFORCE Training Loop (`trainer.py`)

WHY THIS SCRIPT EXISTS:
Implements policy gradient training via REINFORCE with an Exponential Moving Average 
baseline for variance reduction (Kool et al., 2019).

Key Training Steps per Iteration:
1. Sample Instance Generation: Sample a batch of 32 random piece instances on the fly.
2. Autoregressive Rollout: Policy samples piece sequences for each instance in the batch.
3. Environment Execution: Execute sequences in NestingEnv to score scalar rewards R (utilization %).
4. Advantage Calculation: Compute Advantage A = R - b, where b is the moving average baseline.
5. Gradient Backpropagation: Minimize Loss = - mean( log_probs_sum * A ) using Adam optimizer.
6. Benchmark Comparison: Periodically evaluate greedy policy against Phase B heuristic floor (62.62%).
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
from train.moving_average import MovingAverageBaseline


def train_reinforce(
    num_steps: int = 400,
    batch_size: int = 32,
    num_pieces: int = 10,
    lr: float = 1e-4,
    sheet_width: float = 100.0,
    sheet_height: float = 100.0,
    seed: int = 2026,
    checkpoint_path: str = "model/trained_policy.pt",
    plot_path: str = "training_curve.png"
):
    print("=" * 65)
    print("  PHASE D: REINFORCE POLICY GRADIENT TRAINING LOOP")
    print("=" * 65)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Instantiate Policy Network & Optimizer
    policy = AttentionPolicy(
        input_dim=2,
        d_model=128,
        num_heads=8,
        num_layers=2,
        sheet_width=sheet_width,
        sheet_height=sheet_height
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    baseline_tracker = MovingAverageBaseline(beta=0.95)
    env = NestingEnv(sheet_width=sheet_width, sheet_height=sheet_height, num_pieces=num_pieces)

    # Pre-generate a fixed validation set of 50 instances to measure true greedy policy progress
    val_instances = [
        generate_instance(num_pieces=num_pieces, sheet_width=sheet_width, sheet_height=sheet_height, seed=999 + i)
        for i in range(50)
    ]

    # Pre-calculate Phase B Heuristic Floor on validation set
    val_heuristic_scores = [run_largest_first_heuristic(env, inst)[0] for inst in val_instances]
    phase_b_floor = float(np.mean(val_heuristic_scores))

    print(f"Training Parameters:")
    print(f"  Total Training Steps : {num_steps}")
    print(f"  Batch Size           : {batch_size}")
    print(f"  Learning Rate        : {lr}")
    print(f"  Phase B Heuristic    : {phase_b_floor:.2f}% (Benchmark target to beat)")
    print("-" * 65)

    history_steps = []
    history_sample_rewards = []
    history_baseline_vals = []
    history_losses = []
    history_val_greedy = []

    start_time = time.time()

    for step in range(1, num_steps + 1):
        policy.train()

        # Step 1: Generate batch of 32 random rectangle instances
        batch_pieces_list = [
            generate_instance(num_pieces=num_pieces, sheet_width=sheet_width, sheet_height=sheet_height)
            for _ in range(batch_size)
        ]
        batch_pieces_np = np.stack(batch_pieces_list, axis=0)  # Shape: (32, 10, 2)
        batch_tensor = torch.tensor(batch_pieces_np, dtype=torch.float32)

        # Step 2: Forward pass - sample action sequences & track log probabilities
        actions, log_probs_sum, _ = policy(batch_tensor, decode_type="sample")
        actions_np = actions.detach().cpu().numpy()

        # Step 3: Execute sampled actions in NestingEnv to score scalar rewards R (utilization %)
        batch_rewards = []
        for i in range(batch_size):
            state = env.reset(pieces=batch_pieces_np[i])
            for act in actions_np[i]:
                state, reward, done, _ = env.step(act)
            batch_rewards.append(env.score())

        batch_rewards_np = np.array(batch_rewards, dtype=np.float32)
        batch_mean_reward = float(np.mean(batch_rewards_np))

        # Step 4: Calculate Advantage A_i = R_i - b
        advantages_np = baseline_tracker.get_advantage(batch_rewards_np)
        current_baseline = baseline_tracker.update(batch_rewards_np)

        adv_tensor = torch.tensor(advantages_np, dtype=torch.float32, device=log_probs_sum.device)

        # Step 5: REINFORCE Loss calculation & Backprop
        # Loss = - mean( sum(log p(a_t | s_t)) * Advantage )
        loss = - torch.mean(log_probs_sum * adv_tensor)

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent gradient explosion in Transformer attention layers
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        # Record history metrics
        history_steps.append(step)
        history_sample_rewards.append(batch_mean_reward)
        history_baseline_vals.append(current_baseline)
        history_losses.append(loss.item())

        # Step 6: Periodic Validation & Logging (every 20 steps)
        if step % 20 == 0 or step == num_steps:
            policy.eval()
            val_greedy_scores = []
            with torch.no_grad():
                for val_inst in val_instances:
                    val_tensor = torch.tensor(val_inst[np.newaxis, :, :], dtype=torch.float32)
                    greedy_act, _, _ = policy(val_tensor, decode_type="greedy")
                    greedy_seq = greedy_act[0].cpu().numpy()

                    state = env.reset(pieces=val_inst)
                    for act in greedy_seq:
                        state, reward, done, _ = env.step(act)
                    val_greedy_scores.append(env.score())

            mean_val_greedy = float(np.mean(val_greedy_scores))
            history_val_greedy.append((step, mean_val_greedy))

            print(f"Step {step:03d}/{num_steps} | Loss: {loss.item():6.3f} | Sample Utilization: {batch_mean_reward:.2f}% | EMA Baseline: {current_baseline:.2f}% | Greedy Val: {mean_val_greedy:.2f}%")

    elapsed_time = time.time() - start_time
    final_greedy_val = history_val_greedy[-1][1]

    print("-" * 65)
    print("TRAINING COMPLETE!")
    print(f"  Total Execution Time : {elapsed_time:.2f} seconds")
    print(f"  Phase A Random Floor : 60.27%")
    print(f"  Phase B Baseline Floor: {phase_b_floor:.2f}%")
    print(f"  Phase D Trained Greedy: {final_greedy_val:.2f}%")
    print(f"  Gain Over Baseline   : +{final_greedy_val - phase_b_floor:.2f}% percentage points")
    print("-" * 65)

    # Save Trained Model Checkpoint
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({
        'model_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_greedy_val': final_greedy_val,
        'num_steps': num_steps
    }, checkpoint_path)
    print(f"Trained policy checkpoint saved to '{os.path.abspath(checkpoint_path)}'")

    # Save Training Curve Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot 1: Utilization Curves over Training Steps
    ax1.plot(history_steps, history_sample_rewards, label="Sampled Batch Utilization (Training)", color='#1f77b4', alpha=0.35)
    ax1.plot(history_steps, history_baseline_vals, label="EMA Moving Average Baseline (b)", color='#2ca02c', linewidth=2.0)

    val_steps, val_scores = zip(*history_val_greedy)
    ax1.plot(val_steps, val_scores, label="Trained Policy Greedy Validation", color='#d62728', linewidth=2.5, marker='o')

    # Reference Horizontal Lines for Phase A & B
    ax1.axhline(60.27, color='gray', linestyle='--', linewidth=1.5, label="Phase A Random Policy (~60.27%)")
    ax1.axhline(phase_b_floor, color='orange', linestyle='--', linewidth=1.8, label=f"Phase B Heuristic Floor ({phase_b_floor:.2f}%)")

    ax1.set_ylabel("Utilization (%)", fontsize=11, fontweight='bold')
    ax1.set_title("Phase D: REINFORCE Policy Gradient Learning Curve", fontsize=13, fontweight='bold')
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Plot 2: REINFORCE Policy Loss over Time
    ax2.plot(history_steps, history_losses, label="REINFORCE Policy Loss", color='#9467bd', alpha=0.7)
    ax2.set_xlabel("Training Step", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Loss", fontsize=11, fontweight='bold')
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Training curve plot saved successfully to '{os.path.abspath(plot_path)}'\n")
    plt.close()

    return final_greedy_val

if __name__ == "__main__":
    train_reinforce()
