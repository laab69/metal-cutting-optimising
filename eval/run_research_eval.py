"""
Research Benchmark Evaluation Runner (`run_research_eval.py`)

Executes all 5 tiers of the research benchmark suite and generates comprehensive comparison tables and plots:
- Tier 1: Friedman Hard Puzzles (N=5 to 17)
- Tier 3A: Adversarial Heuristic-Trap Benchmark
- Tier 3B: Permutation Invariance & Sensitivity Test
- Tier 3C: Scale Shift Extrapolation (N=10 to N=50)
- Tier 5: Google OR-Tools CP-SAT Exact Global Optimum Solver
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from extensions.rotation_policy import RotationAttentionPolicy
from eval.research_benchmarks import (
    run_tier1_friedman_benchmark,
    run_tier3_heuristic_trap,
    run_tier3_permutation_invariance,
    run_tier3_scale_shift,
    run_tier5_exact_cpsat_solver
)

def main():
    print("=" * 75)
    print("[RESEARCH BENCHMARK] RUNNING EVALUATION STACK (TIERS 1, 3 & 5)")
    print("=" * 75)

    policy = RotationAttentionPolicy(
        input_dim=2,
        d_model=128,
        num_heads=8,
        num_layers=2,
        sheet_width=100.0,
        sheet_height=100.0
    )

    if os.path.exists("model/scaled_policy.pt"):
        ckpt = torch.load("model/scaled_policy.pt", map_location='cpu')
        policy.load_state_dict(ckpt['model_state_dict'], strict=False)
        print("[+] Loaded trained policy weights from 'model/scaled_policy.pt'\n")

    policy.eval()

    # ---------------------------------------------------------------
    # 1. Tier 1: Friedman Known-Optimal Micro-Benchmarks
    # ---------------------------------------------------------------
    print(">>> TIER 1: FRIEDMAN HARD PACKING BENCHMARKS (Exact Bounds)")
    df_friedman = run_tier1_friedman_benchmark(policy)
    print(df_friedman.to_string(index=False))
    print("\n" + "-" * 75)

    # ---------------------------------------------------------------
    # 2. Tier 3A: Heuristic-Trap Adversarial Test
    # ---------------------------------------------------------------
    print(">>> TIER 3A: ADVERSARIAL HEURISTIC-TRAP BENCHMARK")
    res_trap = run_tier3_heuristic_trap(policy)
    for k, v in res_trap.items():
        print(f"   {k:35s}: {v}")
    print("\n" + "-" * 75)

    # ---------------------------------------------------------------
    # 3. Tier 3B: Permutation Invariance Test
    # ---------------------------------------------------------------
    print(">>> TIER 3B: PERMUTATION INVARIANCE & SENSITIVITY TEST (5 Shuffles)")
    res_perm = run_tier3_permutation_invariance(policy, num_shuffles=5)
    for k, v in res_perm.items():
        print(f"   {k:35s}: {v}")
    print("\n" + "-" * 75)

    # ---------------------------------------------------------------
    # 4. Tier 3C: Scale Shift Extrapolation
    # ---------------------------------------------------------------
    print(">>> TIER 3C: SCALE SHIFT / ZERO-SHOT EXTRAPOLATION (N=10 to N=50)")
    df_scale = run_tier3_scale_shift(policy)
    print(df_scale.to_string(index=False))
    print("\n" + "-" * 75)

    # ---------------------------------------------------------------
    # 5. Tier 5: Google OR-Tools CP-SAT Provable Global Optimum
    # ---------------------------------------------------------------
    print(">>> TIER 5: GOOGLE OR-TOOLS CP-SAT EXACT SOLVER BENCHMARK")
    res_cpsat = run_tier5_exact_cpsat_solver(policy, num_pieces=6)
    for k, v in res_cpsat.items():
        print(f"   {k:35s}: {v}")
    print("=" * 75)

    # Save summary plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 9))

    # Plot 1: Friedman Optimality Gaps
    n_vals = df_friedman["Pieces (N)"].astype(int)
    gaps = [float(g.replace('%', '')) for g in df_friedman["Optimality Gap (%)"]]
    ax1.bar(n_vals, gaps, color="#3B82F6", edgecolor="black", alpha=0.85)
    ax1.set_title("Tier 1: Friedman Packing Optimality Gap (%)", fontweight='bold')
    ax1.set_xlabel("Pieces (N)")
    ax1.set_ylabel("Gap to Theoretical Optimum (%)")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Plot 2: Scale Shift Generalization
    scale_n = df_scale["Pieces (N)"].astype(int)
    scale_u = [float(u.replace('%', '')) for u in df_scale["Sheet Utilization (%)"]]
    ax2.plot(scale_n, scale_u, marker='o', color="#10B981", linewidth=2.5, markersize=8)
    ax2.set_title("Tier 3C: Scale Shift Extrapolation (Zero-Shot)", fontweight='bold')
    ax2.set_xlabel("Piece Count (N)")
    ax2.set_ylabel("Sheet Utilization (%)")
    ax2.set_ylim(50, 100)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Plot 3: Heuristic-Trap Comparison
    trap_labels = ["Largest-First", "Attention Policy"]
    trap_vals = [float(res_trap["Heuristic (Largest-First) Util"].replace('%', '')),
                 float(res_trap["Attention Policy Util"].replace('%', ''))]
    ax3.bar(trap_labels, trap_vals, color=["#EF4444", "#8B5CF6"], edgecolor="black", alpha=0.85)
    ax3.set_title("Tier 3A: Adversarial Heuristic-Trap Resistance", fontweight='bold')
    ax3.set_ylabel("Sheet Utilization (%)")
    ax3.grid(True, linestyle="--", alpha=0.5)

    # Plot 4: CP-SAT Global Optimum Comparison
    cpsat_labels = ["Exact CP-SAT (OPT)", "Attention Policy"]
    cpsat_vals = [float(res_cpsat["OR-Tools CP-SAT Provable OPT Util"].replace('%', '')),
                  float(res_cpsat["Neural Attention Policy Util"].replace('%', ''))]
    ax4.bar(cpsat_labels, cpsat_vals, color=["#F59E0B", "#06B6D4"], edgecolor="black", alpha=0.85)
    ax4.set_title("Tier 5: Provable Global Optimum Gap (OR-Tools)", fontweight='bold')
    ax4.set_ylabel("Sheet Utilization (%)")
    ax4.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("eval/research_benchmark_results.png", dpi=150)
    print("\n[+] Saved Research Benchmark Summary Plot to 'eval/research_benchmark_results.png'!")

if __name__ == "__main__":
    main()
