"""
Zero-Shot Generalization Test & Comparative Evaluation (`evaluate_generalization.py`)

WHY THIS SCRIPT IS THE PROJECT DELIVERABLE:
This script performs the definitive evaluation of our Neural Combinatorial Optimization project.

It tests the trained Attention Policy network in a single zero-shot forward pass on 200 
held-out instances it never saw during training.

It benchmarks the trained network against:
1. Random Policy (Phase A)
2. Largest-Piece-First Heuristic (Phase B)
3. Untrained Policy Network (Phase C)
4. Trained Policy Network (Phase D)

It quantifies both Utilization % (solution quality) AND Wall-Clock Latency (computation speed).
"""

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from env.nesting_env import NestingEnv
from baseline.largest_first import run_largest_first_heuristic
from model.policy import AttentionPolicy
from eval.heldout_generator import get_heldout_test_set


def evaluate_all_methods(
    num_instances: int = 200,
    num_pieces: int = 10,
    sheet_width: float = 100.0,
    sheet_height: float = 100.0,
    checkpoint_path: str = "model/trained_policy.pt",
    save_plot_path: str = "eval_generalization_comparison.png"
):
    print("=" * 70)
    print(f"  PHASE E: HELD-OUT GENERALIZATION TEST ({num_instances} UNSEEN INSTANCES)")
    print("=" * 70)

    # 1. Load held-out test instances
    test_instances = get_heldout_test_set(
        num_instances=num_instances,
        num_pieces=num_pieces,
        sheet_width=sheet_width,
        sheet_height=sheet_height,
        seed=5555
    )

    env = NestingEnv(sheet_width=sheet_width, sheet_height=sheet_height, num_pieces=num_pieces)

    # 2. Instantiate and Load Models
    # Trained Policy
    trained_policy = AttentionPolicy(
        input_dim=2,
        d_model=128,
        num_heads=8,
        num_layers=2,
        sheet_width=sheet_width,
        sheet_height=sheet_height
    )
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        trained_policy.load_state_dict(checkpoint['model_state_dict'])
        print(f"[+] Loaded trained model weights from '{checkpoint_path}'")
    else:
        print(f"[!] Warning: Checkpoint '{checkpoint_path}' not found! Running with initial weights.")

    trained_policy.eval()

    # Untrained Policy
    untrained_policy = AttentionPolicy(
        input_dim=2,
        d_model=128,
        num_heads=8,
        num_layers=2,
        sheet_width=sheet_width,
        sheet_height=sheet_height
    )
    untrained_policy.eval()

    # ---------------------------------------------------------
    # METHOD 1: Random Policy Evaluation
    # ---------------------------------------------------------
    print("\nEvaluating Method 1: Random Policy...")
    t0 = time.time()
    random_scores = []
    np.random.seed(5555)
    for inst in test_instances:
        state = env.reset(pieces=inst)
        done = False
        while not done:
            avail = np.where(state["mask"])[0]
            act = int(np.random.choice(avail))
            state, reward, done, _ = env.step(act)
        random_scores.append(env.score())
    t_random = time.time() - t0

    # ---------------------------------------------------------
    # METHOD 2: Largest-Piece-First Heuristic Evaluation
    # ---------------------------------------------------------
    print("Evaluating Method 2: Largest-Piece-First Heuristic...")
    t0 = time.time()
    largest_first_scores = []
    for inst in test_instances:
        score, _ = run_largest_first_heuristic(env, inst)
        largest_first_scores.append(score)
    t_largest_first = time.time() - t0

    # ---------------------------------------------------------
    # METHOD 3: Untrained Policy Network (Greedy)
    # ---------------------------------------------------------
    print("Evaluating Method 3: Untrained Policy Network (Greedy Forward Pass)...")
    t0 = time.time()
    untrained_scores = []
    with torch.no_grad():
        for inst in test_instances:
            inst_tensor = torch.tensor(inst[np.newaxis, :, :], dtype=torch.float32)
            actions, _, _ = untrained_policy(inst_tensor, decode_type="greedy")
            seq = actions[0].cpu().numpy()

            state = env.reset(pieces=inst)
            for act in seq:
                state, reward, done, _ = env.step(act)
            untrained_scores.append(env.score())
    t_untrained = time.time() - t0

    # ---------------------------------------------------------
    # METHOD 4: Trained Attention Policy (Greedy Zero-Shot)
    # ---------------------------------------------------------
    print("Evaluating Method 4: Trained Attention Policy (Zero-Shot Forward Pass)...")
    t0 = time.time()
    trained_scores = []
    sample_trained_seq = None
    with torch.no_grad():
        for idx, inst in enumerate(test_instances):
            inst_tensor = torch.tensor(inst[np.newaxis, :, :], dtype=torch.float32)
            actions, _, _ = trained_policy(inst_tensor, decode_type="greedy")
            seq = actions[0].cpu().numpy()
            if idx == 0:
                sample_trained_seq = list(seq)

            state = env.reset(pieces=inst)
            for act in seq:
                state, reward, done, _ = env.step(act)
            trained_scores.append(env.score())
    t_trained = time.time() - t0

    # ---------------------------------------------------------
    # Compute Statistics
    # ---------------------------------------------------------
    def stats(arr, total_time):
        arr = np.array(arr)
        return float(np.mean(arr)), float(np.std(arr)), (total_time / num_instances) * 1000.0

    mean_rand, std_rand, ms_rand = stats(random_scores, t_random)
    mean_lf, std_lf, ms_lf = stats(largest_first_scores, t_largest_first)
    mean_un, std_un, ms_un = stats(untrained_scores, t_untrained)
    mean_tr, std_tr, ms_tr = stats(trained_scores, t_trained)

    print("\n" + "=" * 70)
    print("  FINAL GENERALIZATION BENCHMARK RESULTS (200 UNSEEN INSTANCES)")
    print("=" * 70)
    print(f"  {'Method':<32} | {'Mean Util (%)':<14} | {'Std Dev (%)':<12} | {'Latency (ms/inst)':<16}")
    print("-" * 70)
    print(f"  {'1. Random Policy':<32} | {mean_rand:6.2f}%         | +/- {std_rand:5.2f}%    | {ms_rand:8.2f} ms")
    print(f"  {'2. Largest-Piece-First Heuristic':<32} | {mean_lf:6.2f}%         | +/- {std_lf:5.2f}%    | {ms_lf:8.2f} ms")
    print(f"  {'3. Untrained Policy Network':<32} | {mean_un:6.2f}%         | +/- {std_un:5.2f}%    | {ms_un:8.2f} ms")
    print(f"  {'4. Trained Attention Policy':<32} | {mean_tr:6.2f}%         | +/- {std_tr:5.2f}%    | {ms_tr:8.2f} ms")
    print("-" * 70)
    print(f"  [+] Gain over Random Policy      : +{mean_tr - mean_rand:.2f}% percentage points")
    print(f"  [+] Gain over Heuristic Floor   : +{mean_tr - mean_lf:.2f}% percentage points")
    print("=" * 70)

    # ---------------------------------------------------------
    # Visual Comparison Plot (3 Subplots)
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(15, 10))

    # Subplot 1: Bar chart comparing average utilization %
    ax1 = plt.subplot(2, 2, 1)
    methods = ["Random Policy", "Largest-First\nHeuristic", "Untrained\nPolicy", "Trained Policy\n(Ours)"]
    means = [mean_rand, mean_lf, mean_un, mean_tr]
    stds = [std_rand, std_lf, std_un, std_tr]
    colors = ['#7f7f7f', '#ff7f0e', '#bcbd22', '#2ca02c']

    bars = ax1.bar(methods, means, yerr=stds, capsize=6, color=colors, edgecolor='black', alpha=0.85)
    ax1.set_ylabel("Average Utilization (%)", fontweight='bold', fontsize=11)
    ax1.set_title("Zero-Shot Quality Comparison (200 Unseen Instances)", fontweight='bold', fontsize=12)
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, h + 2.5, f"{h:.2f}%", ha='center', va='bottom', fontweight='bold')

    # Subplot 2: Inference Latency Comparison
    ax2 = plt.subplot(2, 2, 2)
    latencies = [ms_rand, ms_lf, ms_un, ms_tr]
    bars_lat = ax2.bar(methods, latencies, color=colors, edgecolor='black', alpha=0.85)
    ax2.set_ylabel("Wall-Clock Latency (ms / instance)", fontweight='bold', fontsize=11)
    ax2.set_title("Computation Latency Comparison", fontweight='bold', fontsize=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars_lat:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, h + 0.05, f"{h:.2f} ms", ha='center', va='bottom', fontweight='bold')

    # Subplot 3 & 4: Visual side-by-side layout for Test Instance #0
    sample_inst = test_instances[0]
    # Re-run Largest First layout for sample instance
    score_lf, placed_lf = run_largest_first_heuristic(env, sample_inst)
    # Re-run Trained Policy layout for sample instance
    state = env.reset(pieces=sample_inst)
    for act in sample_trained_seq:
        state, reward, done, _ = env.step(act)
    score_tr = env.score()
    placed_tr = list(env.placed_rects)

    ax3 = plt.subplot(2, 2, 3)
    ax3.add_patch(patches.Rectangle((0, 0), sheet_width, sheet_height, linewidth=2, edgecolor='black', facecolor='#f8f9fa'))
    cmap = plt.cm.tab20(np.linspace(0, 1, num_pieces))
    for idx, (x, y, w, h) in enumerate(placed_lf):
        rect = patches.Rectangle((x, y), w, h, linewidth=1.2, edgecolor='darkorange', facecolor=cmap[idx % len(cmap)], alpha=0.85)
        ax3.add_patch(rect)
        ax3.text(x + w/2.0, y + h/2.0, f"{w:.1f}x{h:.1f}", color='white', weight='bold', fontsize=7, ha='center', va='center')
    ax3.set_xlim(-5, sheet_width + 5)
    ax3.set_ylim(-5, sheet_height + 5)
    ax3.set_aspect('equal')
    ax3.set_title(f"Heuristic Layout (Largest-First)\nUtilization: {score_lf:.2f}%", fontweight='bold', fontsize=10)

    ax4 = plt.subplot(2, 2, 4)
    ax4.add_patch(patches.Rectangle((0, 0), sheet_width, sheet_height, linewidth=2, edgecolor='black', facecolor='#f8f9fa'))
    for idx, (x, y, w, h) in enumerate(placed_tr):
        rect = patches.Rectangle((x, y), w, h, linewidth=1.2, edgecolor='darkgreen', facecolor=cmap[idx % len(cmap)], alpha=0.85)
        ax4.add_patch(rect)
        ax4.text(x + w/2.0, y + h/2.0, f"{w:.1f}x{h:.1f}", color='white', weight='bold', fontsize=7, ha='center', va='center')
    ax4.set_xlim(-5, sheet_width + 5)
    ax4.set_ylim(-5, sheet_height + 5)
    ax4.set_aspect('equal')
    ax4.set_title(f"Trained Policy Layout (Zero-Shot Ours)\nUtilization: {score_tr:.2f}%", fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_plot_path, dpi=150)
    print(f"\nGeneralization comparison plot saved successfully to '{os.path.abspath(save_plot_path)}'\n")
    plt.close()

    return mean_tr

if __name__ == "__main__":
    evaluate_all_methods()
