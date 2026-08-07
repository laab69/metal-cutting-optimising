"""
Scaled Policy Benchmark Evaluation (`evaluate_scaled.py`)

WHY THIS SCRIPT EXISTS:
Evaluates the scaled Attention Policy network (trained with Kool et al.'s Rollout Baseline) 
on 100 held-out unseen N=20 instances to demonstrate a clear advantage over classical heuristics.
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


def evaluate_scaled_generalization(
    num_instances: int = 100,
    num_pieces: int = 20,
    sheet_width: float = 100.0,
    sheet_height: float = 100.0,
    checkpoint_path: str = "model/scaled_policy.pt",
    save_plot_path: str = "scaled_generalization_benchmark.png"
):
    print("=" * 70)
    print(f"  SCALED GENERALIZATION BENCHMARK (N = {num_pieces} PIECES, {num_instances} UNSEEN INSTANCES)")
    print("=" * 70)

    # 1. Generate 100 held-out N=20 test instances
    test_instances = [
        generate_instance(num_pieces=num_pieces, sheet_width=sheet_width, sheet_height=sheet_height, seed=6666 + i)
        for i in range(num_instances)
    ]

    env = NestingEnv(sheet_width=sheet_width, sheet_height=sheet_height, num_pieces=num_pieces)

    # 2. Load Scaled Trained Policy
    policy = AttentionPolicy(
        input_dim=2,
        d_model=128,
        num_heads=8,
        num_layers=2,
        sheet_width=sheet_width,
        sheet_height=sheet_height
    )
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        policy.load_state_dict(checkpoint['model_state_dict'])
        print(f"[+] Loaded scaled model weights from '{checkpoint_path}'")
    else:
        print(f"[!] Warning: Checkpoint '{checkpoint_path}' not found!")

    policy.eval()

    # 3. Evaluate Random Policy
    print("\n1. Evaluating Random Policy...")
    t0 = time.time()
    random_scores = []
    np.random.seed(6666)
    for inst in test_instances:
        state = env.reset(pieces=inst)
        done = False
        while not done:
            avail = np.where(state["mask"])[0]
            act = int(np.random.choice(avail))
            state, reward, done, _ = env.step(act)
        random_scores.append(env.score())
    t_rand = time.time() - t0

    # 4. Evaluate Largest-Piece-First Heuristic
    print("2. Evaluating Largest-Piece-First Heuristic...")
    t0 = time.time()
    largest_first_scores = []
    for inst in test_instances:
        score, _ = run_largest_first_heuristic(env, inst)
        largest_first_scores.append(score)
    t_lf = time.time() - t0

    # 5. Evaluate Scaled Attention Policy (Greedy Zero-Shot)
    print("3. Evaluating Scaled Attention Policy (Ours)...")
    t0 = time.time()
    policy_scores = []
    with torch.no_grad():
        for inst in test_instances:
            inst_t = torch.tensor(inst[np.newaxis, :, :], dtype=torch.float32)
            actions, _, _ = policy(inst_t, decode_type="greedy")
            seq = actions[0].cpu().numpy()

            state = env.reset(pieces=inst)
            for act in seq:
                state, reward, done, _ = env.step(act)
            policy_scores.append(env.score())
    t_pol = time.time() - t0

    # Compute Statistics
    def stats(arr, total_time):
        arr = np.array(arr)
        return float(np.mean(arr)), float(np.std(arr)), (total_time / num_instances) * 1000.0

    mean_rand, std_rand, ms_rand = stats(random_scores, t_rand)
    mean_lf, std_lf, ms_lf = stats(largest_first_scores, t_lf)
    mean_pol, std_pol, ms_pol = stats(policy_scores, t_pol)

    print("\n" + "=" * 70)
    print(f"  SCALED GENERALIZATION RESULTS SUMMARY (N = {num_pieces} PIECES)")
    print("=" * 70)
    print(f"  {'Method':<32} | {'Mean Util (%)':<14} | {'Std Dev (%)':<12} | {'Latency (ms/inst)':<16}")
    print("-" * 70)
    print(f"  {'1. Random Policy':<32} | {mean_rand:6.2f}%         | +/- {std_rand:5.2f}%    | {ms_rand:8.2f} ms")
    print(f"  {'2. Largest-Piece-First Heuristic':<32} | {mean_lf:6.2f}%         | +/- {std_lf:5.2f}%    | {ms_lf:8.2f} ms")
    print(f"  {'3. Scaled Attention Policy (Ours)':<32} | {mean_pol:6.2f}%         | +/- {std_pol:5.2f}%    | {ms_pol:8.2f} ms")
    print("-" * 70)
    print(f"  [+] Gain over Random Policy       : +{mean_pol - mean_rand:.2f}% percentage points")
    print(f"  [+] Advantage Gain over Heuristic: +{mean_pol - mean_lf:.2f}% percentage points")
    print("=" * 70)

    # Comparison Bar Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    methods = ["Random Policy", "Largest-First\nHeuristic", "Scaled Policy\n(Ours, N=20)"]
    means = [mean_rand, mean_lf, mean_pol]
    stds = [std_rand, std_lf, std_pol]
    colors = ['#7f7f7f', '#ff7f0e', '#2ca02c']

    bars = ax.bar(methods, means, yerr=stds, capsize=8, color=colors, edgecolor='black', alpha=0.85)
    ax.set_ylabel("Average Utilization (%)", fontweight='bold', fontsize=11)
    ax.set_title(f"Scaled NCO Benchmark (N={num_pieces} Pieces) over 100 Unseen Instances", fontweight='bold', fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, h + 2.5, f"{h:.2f}%", ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_plot_path, dpi=150)
    print(f"Scaled generalization plot saved to '{os.path.abspath(save_plot_path)}'\n")
    plt.close()

    return mean_pol

if __name__ == "__main__":
    evaluate_scaled_generalization()
