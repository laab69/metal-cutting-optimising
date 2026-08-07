"""
Baseline Evaluation Script (`evaluate_baseline.py`)

WHY THIS SCRIPT EXISTS:
To obtain statistically meaningful benchmark numbers, evaluating on a single instance is 
insufficient due to random variance in piece sizes. 

This script evaluates both the Random Policy and the Largest-Piece-First heuristic across 
100 randomly generated instances, establishing the target benchmark floor (average utilization %) 
that our neural policy will be trained to beat.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from env.generator import generate_instance
from env.nesting_env import NestingEnv
from baseline.largest_first import run_largest_first_heuristic


def evaluate_baselines(
    num_instances: int = 100,
    num_pieces: int = 10,
    sheet_width: float = 100.0,
    sheet_height: float = 100.0,
    seed: int = 2026,
    save_plot_path: str = "baseline_comparison.png"
):
    print("=" * 65)
    print(f"  PHASE B: EVALUATING BASELINE HEURISTIC OVER {num_instances} INSTANCES")
    print("=" * 65)

    np.random.seed(seed)

    env = NestingEnv(sheet_width=sheet_width, sheet_height=sheet_height, num_pieces=num_pieces)

    random_utilizations = []
    largest_first_utilizations = []

    start_time = time.time()

    # Pre-generate 100 instances so both policies evaluate on the exact same problem sets
    test_instances = [
        generate_instance(num_pieces=num_pieces, sheet_width=sheet_width, sheet_height=sheet_height, seed=seed + i)
        for i in range(num_instances)
    ]

    # Evaluate Random Policy
    for instances_pieces in test_instances:
        state = env.reset(pieces=instances_pieces)
        done = False
        while not done:
            avail = np.where(state["mask"])[0]
            action = int(np.random.choice(avail))
            state, reward, done, _ = env.step(action)
        random_utilizations.append(env.score())

    # Evaluate Largest-Piece-First Heuristic
    for instances_pieces in test_instances:
        util, _ = run_largest_first_heuristic(env, instances_pieces)
        largest_first_utilizations.append(util)

    total_time = time.time() - start_time

    # Calculate statistics
    avg_random = float(np.mean(random_utilizations))
    std_random = float(np.std(random_utilizations))

    avg_largest_first = float(np.mean(largest_first_utilizations))
    std_largest_first = float(np.std(largest_first_utilizations))

    improvement = avg_largest_first - avg_random

    print("\nBENCHMARK RESULTS SUMMARY:")
    print("-" * 65)
    print(f"  Total Evaluation Instances : {num_instances}")
    print(f"  Pieces Per Instance        : {num_pieces}")
    print(f"  Sheet Dimensions           : {sheet_width:.1f} x {sheet_height:.1f}")
    print(f"  Total Wall-Clock Time      : {total_time:.3f} seconds ({total_time/num_instances*1000:.2f} ms/instance)")
    print("-" * 65)
    print(f"  1. Random Policy           : {avg_random:.2f}% +/- {std_random:.2f}%")
    print(f"  2. Largest-Piece-First     : {avg_largest_first:.2f}% +/- {std_largest_first:.2f}%")
    print("-" * 65)
    print(f"  [+] Baseline Benchmark Floor: {avg_largest_first:.2f}% utilization")
    print(f"  [+] Heuristic Gain over Random: +{improvement:.2f}% percentage points")
    print("=" * 65)

    # Generate Comparison Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Bar chart comparing average utilization %
    bars = ax1.bar(
        ["Random Policy\n(Phase A)", "Largest-Piece-First\n(Phase B Baseline Floor)"],
        [avg_random, avg_largest_first],
        yerr=[std_random, std_largest_first],
        capsize=8,
        color=['#7f7f7f', '#2ca02c'],
        edgecolor='black',
        alpha=0.85
    )
    ax1.set_ylabel("Average Utilization (%)", fontsize=11, fontweight='bold')
    ax1.set_title(f"Baseline Benchmark Performance over {num_instances} Instances", fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # Add numeric labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, height + 3, f"{height:.2f}%", ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Subplot 2: Side-by-side visual comparison for instance #0
    sample_instance = test_instances[0]
    # Run Random Policy on sample instance
    state = env.reset(pieces=sample_instance)
    np.random.seed(seed)
    done = False
    while not done:
        avail = np.where(state["mask"])[0]
        action = int(np.random.choice(avail))
        state, reward, done, _ = env.step(action)
    rand_placed = list(env.placed_rects)
    rand_score = env.score()

    # Run Largest First on sample instance
    lf_score, lf_placed = run_largest_first_heuristic(env, sample_instance)

    # Plot Largest First Layout on ax2
    ax2.add_patch(patches.Rectangle((0, 0), sheet_width, sheet_height, linewidth=2, edgecolor='black', facecolor='#f8f9fa'))
    colors = plt.cm.tab20(np.linspace(0, 1, num_pieces))
    for idx, (x, y, w, h) in enumerate(lf_placed):
        color = colors[idx % len(colors)]
        rect = patches.Rectangle((x, y), w, h, linewidth=1.2, edgecolor='darkgreen', facecolor=color, alpha=0.85)
        ax2.add_patch(rect)
        ax2.text(x + w / 2.0, y + h / 2.0, f"{w:.1f}x{h:.1f}", color='white', weight='bold', fontsize=8, ha='center', va='center')

    ax2.set_xlim(-5, sheet_width + 5)
    ax2.set_ylim(-5, sheet_height + 5)
    ax2.set_aspect('equal')
    ax2.set_title(f"Sample Instance #0 Layout (Largest-Piece-First)\nUtilization: {lf_score:.2f}% (vs Random: {rand_score:.2f}%)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("X (Sheet Width)")
    ax2.set_ylabel("Y (Sheet Height)")
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_plot_path, dpi=150)
    print(f"Comparison plot saved successfully to '{os.path.abspath(save_plot_path)}'\n")
    plt.close()

    return avg_largest_first

if __name__ == "__main__":
    evaluate_baselines()
