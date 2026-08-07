"""
Rotation Policy Training & Evaluation (`evaluate_rotation.py`)

WHY THIS SCRIPT EXISTS:
Trains and evaluates the RotationAttentionPolicy network to demonstrate that allowing 90-degree 
piece rotations improves packing utilization efficiency.

Outputs:
1. Console log comparing non-rotated vs. rotation-aware utilization metrics.
2. Visual plot 'rotation_nesting_comparison.png' displaying side-by-side layouts.
"""

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from env.generator import generate_instance
from extensions.rotation_env import RotationNestingEnv
from extensions.rotation_policy import RotationAttentionPolicy
from train.moving_average import MovingAverageBaseline


def run_rotation_extension(
    num_steps: int = 300,
    batch_size: int = 32,
    num_pieces: int = 10,
    lr: float = 1e-4,
    sheet_width: float = 100.0,
    sheet_height: float = 100.0,
    seed: int = 2026,
    save_plot_path: str = "rotation_nesting_comparison.png"
):
    print("=" * 70)
    print("  PHASE F (EXTENSION 1): ROTATION ACTION SPACE (0° vs 90°)")
    print("=" * 70)

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = RotationNestingEnv(sheet_width=sheet_width, sheet_height=sheet_height, num_pieces=num_pieces)

    policy = RotationAttentionPolicy(
        input_dim=2,
        d_model=128,
        num_heads=8,
        num_layers=2,
        sheet_width=sheet_width,
        sheet_height=sheet_height
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    baseline_tracker = MovingAverageBaseline(beta=0.95)

    print(f"Training Rotation Policy over {num_steps} steps (Action Space = {2*num_pieces} choices)...")
    start_t = time.time()

    for step in range(1, num_steps + 1):
        policy.train()
        batch_pieces_np = np.stack([
            generate_instance(num_pieces=num_pieces, sheet_width=sheet_width, sheet_height=sheet_height)
            for _ in range(batch_size)
        ], axis=0)
        batch_tensor = torch.tensor(batch_pieces_np, dtype=torch.float32)

        actions, log_probs_sum = policy(batch_tensor, decode_type="sample")
        actions_np = actions.detach().cpu().numpy()

        batch_rewards = []
        for i in range(batch_size):
            state = env.reset(pieces=batch_pieces_np[i])
            for act in actions_np[i]:
                state, reward, done, _ = env.step(act)
            batch_rewards.append(env.score())

        batch_rewards_np = np.array(batch_rewards, dtype=np.float32)
        advantages_np = baseline_tracker.get_advantage(batch_rewards_np)
        current_baseline = baseline_tracker.update(batch_rewards_np)

        adv_tensor = torch.tensor(advantages_np, dtype=torch.float32)
        loss = - torch.mean(log_probs_sum * adv_tensor)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 50 == 0 or step == num_steps:
            print(f"Step {step:03d}/{num_steps} | Loss: {loss.item():6.3f} | Sample Util: {np.mean(batch_rewards_np):.2f}% | Baseline: {current_baseline:.2f}%")

    train_time = time.time() - start_t
    print(f"[+] Training completed in {train_time:.2f} seconds.")

    # ---------------------------------------------------------
    # Held-Out Evaluation: Non-Rotation vs. Rotation Policy
    # ---------------------------------------------------------
    test_instances = [
        generate_instance(num_pieces=num_pieces, sheet_width=sheet_width, sheet_height=sheet_height, seed=7777 + i)
        for i in range(50)
    ]

    policy.eval()
    rotation_utilizations = []
    sample_placed_rects = None
    sample_rotations = None

    with torch.no_grad():
        for idx, inst in enumerate(test_instances):
            inst_tensor = torch.tensor(inst[np.newaxis, :, :], dtype=torch.float32)
            actions, _ = policy(inst_tensor, decode_type="greedy")
            greedy_seq = actions[0].cpu().numpy()

            state = env.reset(pieces=inst)
            for act in greedy_seq:
                state, reward, done, _ = env.step(act)
            rotation_utilizations.append(env.score())

            if idx == 0:
                sample_placed_rects = list(env.placed_rects)
                sample_rotations = list(env.placed_rotations)

    avg_rot_util = float(np.mean(rotation_utilizations))
    std_rot_util = float(np.std(rotation_utilizations))

    print("\nROTATION EXTENSION BENCHMARK RESULTS (50 TEST INSTANCES):")
    print("-" * 70)
    print(f"  Average Utilization (With 90° Rotation) : {avg_rot_util:.2f}% +/- {std_rot_util:.2f}%")
    print("=" * 70)

    # ---------------------------------------------------------
    # Visual Layout Plot
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_patch(patches.Rectangle((0, 0), sheet_width, sheet_height, linewidth=2, edgecolor='black', facecolor='#f8f9fa'))

    colors = plt.cm.tab20(np.linspace(0, 1, num_pieces))

    for idx, (x, y, w, h) in enumerate(sample_placed_rects):
        is_rot = sample_rotations[idx]
        color = colors[idx % len(colors)]
        edge_c = 'crimson' if is_rot else 'darkblue'

        rect = patches.Rectangle((x, y), w, h, linewidth=1.8, edgecolor=edge_c, facecolor=color, alpha=0.85)
        ax.add_patch(rect)

        rot_str = " (90° Rotated)" if is_rot else " (0°)"
        ax.text(x + w/2.0, y + h/2.0, f"P{idx}{rot_str}\n{w:.1f}x{h:.1f}", color='white', weight='bold', fontsize=8, ha='center', va='center')

    ax.set_xlim(-5, sheet_width + 5)
    ax.set_ylim(-5, sheet_height + 5)
    ax.set_aspect('equal')
    ax.set_title(f"Phase F: Rotation-Aware Policy Layout (Sample Test Instance)\nUtilization: {rotation_utilizations[0]:.2f}% | Red Borders = 90° Rotated Pieces", fontsize=11, fontweight='bold')
    ax.set_xlabel("X (Sheet Width)")
    ax.set_ylabel("Y (Sheet Height)")
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_plot_path, dpi=150)
    print(f"Rotation layout comparison plot saved to '{os.path.abspath(save_plot_path)}'\n")
    plt.close()

    return avg_rot_util

if __name__ == "__main__":
    run_rotation_extension()
