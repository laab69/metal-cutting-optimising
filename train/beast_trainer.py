import os
import sys
import time

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.optim as optim
from scipy import stats
import matplotlib.pyplot as plt

from extensions.rotation_env import MultiAngleNestingEnv
from extensions.rotation_policy import RotationAttentionPolicy


def evaluate_policy_on_dataset(policy: RotationAttentionPolicy, instances: list, sheet_w=100.0, sheet_h=100.0) -> float:
    policy.eval()
    scores = []

    for pieces in instances:
        num_pieces = len(pieces)
        env = MultiAngleNestingEnv(sheet_width=sheet_w, sheet_height=sheet_h, num_pieces=num_pieces)
        env.reset(pieces=pieces)

        batch_pieces = torch.tensor(pieces[np.newaxis, :, :], dtype=torch.float32)
        with torch.no_grad():
            actions_t, _ = policy(batch_pieces, decode_type="greedy")
            actions = actions_t[0].cpu().numpy()

        for act in actions:
            state, reward, done, info = env.step(act)
            if done:
                scores.append(reward)
                break

    return float(np.mean(scores))


def train_beast_policy(
    num_iterations: int = 3000,
    batch_size: int = 64,
    min_pieces: int = 10,
    max_pieces: int = 30,
    lr: float = 1e-4,
    eval_freq: int = 50,
    checkpoint_freq: int = 250,
    save_path: str = "model/beast_policy.pt"
):
    print("=" * 70)
    print("[LAUNCH] INDUSTRIAL BEAST-MODE REINFORCE TRAINING (Kool et al. 2019)")
    print(f"   Batch Size: {batch_size} | Total Steps: {num_iterations}")
    print(f"   Piece Count per Instance: {min_pieces} to {max_pieces} pieces")
    print(f"   Device: {'CUDA GPU' if torch.cuda.is_available() else 'CPU (Multi-Threaded)'}")
    print("=" * 70)

    os.makedirs("model", exist_ok=True)
    os.makedirs("train", exist_ok=True)

    # Use max pieces for fixed tensor size initialization
    policy = RotationAttentionPolicy(
        input_dim=2,
        d_model=128,
        num_heads=8,
        num_layers=2,
        sheet_width=100.0,
        sheet_height=100.0
    )

    baseline_policy = RotationAttentionPolicy(
        input_dim=2,
        d_model=128,
        num_heads=8,
        num_layers=2,
        sheet_width=100.0,
        sheet_height=100.0
    )

    # Load existing checkpoint if available
    if os.path.exists("model/scaled_policy.pt"):
        ckpt = torch.load("model/scaled_policy.pt", map_location='cpu')
        policy.load_state_dict(ckpt['model_state_dict'], strict=False)
        print("Successfully pre-loaded existing weights from model/scaled_policy.pt!")

    baseline_policy.load_state_dict(policy.state_dict())
    baseline_policy.eval()

    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # 100 Held-Out Evaluation Instances
    np.random.seed(9999)
    eval_instances = [
        np.random.uniform(5.0, 35.0, size=(np.random.randint(min_pieces, max_pieces + 1), 2)).astype(np.float32)
        for _ in range(20)
    ]

    print("[+] Computing Initial Baseline Utilization Floor on Validation Dataset...", flush=True)
    baseline_eval_score = evaluate_policy_on_dataset(baseline_policy, eval_instances)
    print(f"[+] Initial Baseline Utilization Floor: {baseline_eval_score:.2f}%\n", flush=True)

    history_candidate = []
    history_baseline = []
    history_loss = []

    start_time = time.time()

    for it in range(1, num_iterations + 1):
        policy.train()

        # Sample dynamic N pieces for this batch
        N = np.random.randint(min_pieces, max_pieces + 1)
        batch_pieces_np = np.random.uniform(5.0, 35.0, size=(batch_size, N, 2)).astype(np.float32)
        batch_pieces_t = torch.tensor(batch_pieces_np, dtype=torch.float32)

        # Candidate sampling rollout
        candidate_actions, candidate_log_probs = policy(batch_pieces_t, decode_type="sample")

        # Rollout rewards for candidate
        candidate_rewards = []
        for b in range(batch_size):
            env = MultiAngleNestingEnv(sheet_width=100.0, sheet_height=100.0, num_pieces=N)
            env.reset(pieces=batch_pieces_np[b])
            for act in candidate_actions[b].cpu().numpy():
                state, r, done, info = env.step(act)
                if done:
                    candidate_rewards.append(r)
                    break

        candidate_rewards_t = torch.tensor(candidate_rewards, dtype=torch.float32)

        # Rollout rewards for baseline (Greedy)
        with torch.no_grad():
            baseline_actions, _ = baseline_policy(batch_pieces_t, decode_type="greedy")
            baseline_rewards = []
            for b in range(batch_size):
                env = MultiAngleNestingEnv(sheet_width=100.0, sheet_height=100.0, num_pieces=N)
                env.reset(pieces=batch_pieces_np[b])
                for act in baseline_actions[b].cpu().numpy():
                    state, r, done, info = env.step(act)
                    if done:
                        baseline_rewards.append(r)
                        break
            baseline_rewards_t = torch.tensor(baseline_rewards, dtype=torch.float32)

        # REINFORCE Advantage
        advantage = candidate_rewards_t - baseline_rewards_t
        loss = - (advantage * candidate_log_probs).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        history_loss.append(loss.item())

        # Real-time per-step progress streaming on EVERY iteration!
        print(f"Step [{it:4d}/{num_iterations}] | Loss: {loss.item():.4f} | Batch Avg Utilization: {candidate_rewards_t.mean():.2f}% | Piece Count N={N}", flush=True)

        # Evaluate and test baseline update
        if it % eval_freq == 0:
            cand_score = evaluate_policy_on_dataset(policy, eval_instances)
            base_score = evaluate_policy_on_dataset(baseline_policy, eval_instances)

            history_candidate.append((it, cand_score))
            history_baseline.append((it, base_score))

            # Calculate paired t-test over evaluation instances
            cand_scores_list = []
            base_scores_list = []
            for p_inst in eval_instances:
                c_s = evaluate_policy_on_dataset(policy, [p_inst])
                b_s = evaluate_policy_on_dataset(baseline_policy, [p_inst])
                cand_scores_list.append(c_s)
                base_scores_list.append(b_s)

            t_stat, p_val = stats.ttest_rel(cand_scores_list, base_scores_list)

            updated = False
            if cand_score > base_score and p_val < 0.05:
                baseline_policy.load_state_dict(policy.state_dict())
                updated = True

            elapsed_min = (time.time() - start_time) / 60.0
            print(f"--- BENCHMARK EVAL Step [{it:4d}/{num_iterations}] | Candidate: {cand_score:.2f}% | Baseline: {base_score:.2f}% | p-val: {p_val:.4f} | Updated: {updated} | Time: {elapsed_min:.1f}m ---", flush=True)

        # Save Checkpoint
        if it % checkpoint_freq == 0 or it == num_iterations:
            ckpt_dict = {
                'iteration': it,
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'candidate_history': history_candidate,
                'baseline_history': history_baseline
            }
            torch.save(ckpt_dict, save_path)
            torch.save(ckpt_dict, "model/scaled_policy.pt")
            print(f"   [+] Saved Checkpoint to {save_path} and model/scaled_policy.pt!")

            # Plot live training curve
            fig, ax = plt.subplots(figsize=(10, 5))
            if history_candidate:
                its, c_vals = zip(*history_candidate)
                _, b_vals = zip(*history_baseline)
                ax.plot(its, c_vals, label="Candidate Neural Policy", color="#2563EB", linewidth=2)
                ax.plot(its, b_vals, label="Greedy Rollout Baseline", color="#DC2626", linestyle="--", linewidth=2)
            ax.set_title("Beast-Mode Industrial Training Curve (Kool et al., 2019)", fontsize=12, fontweight='bold')
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Sheet Utilization %")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend()
            plt.tight_layout()
            plt.savefig("train/beast_training_curve.png", dpi=150)
            plt.close(fig)

    print("=" * 70)
    print("[SUCCESS] BEAST-MODE TRAINING COMPLETE!")
    print(f"   Final Checkpoint Saved: {save_path}")
    print("=" * 70)


if __name__ == "__main__":
    train_beast_policy(
        num_iterations=3000,
        batch_size=16,
        min_pieces=10,
        max_pieces=30,
        lr=1e-4,
        eval_freq=50,
        checkpoint_freq=250
    )
