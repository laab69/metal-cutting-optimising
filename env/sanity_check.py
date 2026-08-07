"""
Random Policy Sanity Check & Layout Visualizer

WHY THIS SCRIPT EXISTS:
Before building neural networks or RL training loops, standard software engineering in AI 
requires validating the environment and physical simulation plumbing first.

This script executes a *Random Policy* (choosing unplaced pieces uniformly at random) to:
1. Confirm that `NestingEnv` and `place_bottom_left` produce non-overlapping valid layouts.
2. Establish a baseline numeric score for pure random selection.
3. Render a visual layout plot so we can visually inspect piece placement on the metal sheet.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from env.nesting_env import NestingEnv

def run_random_policy_sanity_check(
    num_pieces: int = 10,
    sheet_width: float = 100.0,
    sheet_height: float = 100.0,
    seed: int = 42,
    save_path: str = "env_sanity_check.png"
):
    print("=" * 60)
    print("  PHASE A: ENVIRONMENT & PLUMBING SANITY RUN (RANDOM POLICY)")
    print("=" * 60)

    # 1. Initialize environment
    env = NestingEnv(sheet_width=sheet_width, sheet_height=sheet_height, num_pieces=num_pieces)
    state = env.reset(seed=seed)

    print(f"Generated instance with {num_pieces} rectangular pieces.")
    print(f"Sheet dimensions: {sheet_width:.1f} x {sheet_height:.1f} (Area = {env.sheet_area:.1f})")
    print("\nPiece list [width, height]:")
    for i, (w, h) in enumerate(state["pieces"]):
        print(f"  Piece {i}: {w:.1f} x {h:.1f} (Area = {w*h:.1f})")

    # 2. Run episode using Random Policy
    np.random.seed(seed)
    done = False
    episode_step = 0

    while not done:
        # Get indices of pieces that are still available (mask == True)
        available_indices = np.where(state["mask"])[0]

        # Action selection: pick one available piece uniformly at random
        action_idx = int(np.random.choice(available_indices))

        state, reward, done, info = env.step(action_idx)
        episode_step += 1

        placed_status = "Placed" if info["placed_successfully"] else "Fitted/Skipped (No space)"
        print(f"Step {episode_step:02d} | Picked Piece {action_idx:02d} ({state['pieces'][action_idx][0]:.1f}x{state['pieces'][action_idx][1]:.1f}) -> {placed_status}")

    final_utilization = env.score()
    placed_count = len(env.placed_rects)

    print("-" * 60)
    print("SANITY RUN RESULTS:")
    print(f"  Total Pieces Placed: {placed_count} / {num_pieces}")
    print(f"  Final Utilization  : {final_utilization:.2f}%")
    print("-" * 60)

    # 3. Visualize layout with Matplotlib
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw metal sheet border
    sheet_rect = patches.Rectangle((0, 0), sheet_width, sheet_height, linewidth=2, edgecolor='black', facecolor='#f0f0f0')
    ax.add_patch(sheet_rect)

    # Color map for placed pieces
    colors = plt.cm.tab20(np.linspace(0, 1, num_pieces))

    for idx, (x, y, w, h) in enumerate(env.placed_rects):
        piece_orig_idx = env.placed_indices[idx]
        color = colors[piece_orig_idx % len(colors)]

        # Draw piece rectangle
        rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='darkblue', facecolor=color, alpha=0.8)
        ax.add_patch(rect)

        # Label piece index and dimensions
        cx, cy = x + w / 2.0, y + h / 2.0
        ax.text(cx, cy, f"P{piece_orig_idx}\n{w:.1f}x{h:.1f}", color='white', weight='bold', fontsize=9, ha='center', va='center')

    ax.set_xlim(-5, sheet_width + 5)
    ax.set_ylim(-5, sheet_height + 5)
    ax.set_aspect('equal')
    ax.set_title(f"Phase A Sanity Run (Random Policy)\nPlaced: {placed_count}/{num_pieces} | Utilization: {final_utilization:.2f}%", fontsize=12, fontweight='bold')
    ax.set_xlabel("X (Sheet Width)")
    ax.set_ylabel("Y (Sheet Height)")
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Layout plot saved successfully to '{os.path.abspath(save_path)}'")
    plt.close()

    return final_utilization

if __name__ == "__main__":
    run_random_policy_sanity_check()
