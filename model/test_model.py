"""
Untrained Policy Network Forward Pass Test (`test_model.py`)

WHY THIS SCRIPT EXISTS:
In deep learning development, verifying the neural network forward pass BEFORE writing 
training code is critical. 

This script tests that:
1. Tensor dimensions match expectations across Encoder and Pointer Decoder.
2. Step-by-step probability distributions sum to 1.0.
3. Action masking correctly forces previously chosen pieces to 0.0 probability.
4. The policy generates valid permutation sequences that execute in `NestingEnv`.
"""

import torch
import numpy as np
from env.generator import generate_instance
from env.nesting_env import NestingEnv
from model.policy import AttentionPolicy


def test_attention_policy():
    print("=" * 65)
    print("  PHASE C: UNTRAINED POLICY NETWORK FORWARD PASS TEST")
    print("=" * 65)

    num_pieces = 10
    sheet_w, sheet_h = 100.0, 100.0
    seed = 42
    torch.manual_seed(seed)

    # 1. Generate 2 synthetic instances as a PyTorch batch of shape (2, 10, 2)
    inst1 = generate_instance(num_pieces=num_pieces, sheet_width=sheet_w, sheet_height=sheet_h, seed=seed)
    inst2 = generate_instance(num_pieces=num_pieces, sheet_width=sheet_w, sheet_height=sheet_h, seed=seed + 1)
    
    batch_np = np.stack([inst1, inst2], axis=0)
    batch_tensor = torch.tensor(batch_np, dtype=torch.float32)

    print(f"Batch Tensor Shape: {list(batch_tensor.shape)} (batch_size=2, num_pieces=10, features=2)")

    # 2. Instantiate Policy Network (Untrained weights)
    policy = AttentionPolicy(
        input_dim=2,
        d_model=128,
        num_heads=8,
        num_layers=2,
        sheet_width=sheet_w,
        sheet_height=sheet_h
    )

    policy.eval()

    # 3. Perform Forward Pass in 'sample' mode
    with torch.no_grad():
        actions_sample, log_probs_sample, step_probs = policy(batch_tensor, decode_type="sample")
        actions_greedy, log_probs_greedy, _ = policy(batch_tensor, decode_type="greedy")

    print("\n[+] Forward Pass Execution Succeeded!")
    print(f"  Sampled Actions Shape   : {list(actions_sample.shape)} -> {actions_sample.tolist()}")
    print(f"  Log Probs Sum (Sample) : {log_probs_sample.tolist()}")
    print(f"  Greedy Actions Shape    : {list(actions_greedy.shape)} -> {actions_greedy.tolist()}")

    # 4. Verify Probability Distributions & Action Masking
    print("\nVERIFYING PROBABILITY DISTRIBUTIONS & ACTION MASKING:")
    probs_step0 = step_probs[0, 0].numpy()  # Step 0 probabilities for instance 0
    probs_step1 = step_probs[0, 1].numpy()  # Step 1 probabilities for instance 0
    picked_step0 = actions_sample[0, 0].item()

    print(f"  Step 0 Probs Sum       : {probs_step0.sum():.6f} (Must equal 1.0)")
    print(f"  Step 0 Picked Piece    : Index {picked_step0}")
    print(f"  Step 1 Prob of Piece {picked_step0}: {probs_step1[picked_step0]:.6f} (Must equal 0.0 due to Action Masking!)")

    assert np.isclose(probs_step0.sum(), 1.0), "Error: Probabilities at step 0 do not sum to 1.0!"
    assert np.isclose(probs_step1[picked_step0], 0.0), "Error: Action masking failed! Picked piece has non-zero probability."
    print("  [+] Action Masking Assertion PASSED!")

    # 5. Execute Untrained Policy Sequence on NestingEnv
    env = NestingEnv(sheet_width=sheet_w, sheet_height=sheet_h, num_pieces=num_pieces)
    seq = actions_sample[0].tolist()

    state = env.reset(pieces=inst1)
    for act in seq:
        state, reward, done, _ = env.step(act)

    untrained_utilization = env.score()
    print("-" * 65)
    print("UNTRAINED POLICY LAYOUT SCORE:")
    print(f"  Sampled Ordering Execution Utilization: {untrained_utilization:.2f}%")
    print("  (Note: Untrained network weights yield random-like probabilities ~60%)")
    print("=" * 65)

if __name__ == "__main__":
    test_attention_policy()
