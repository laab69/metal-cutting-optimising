
# Phase C Complete! 🎉

Here is a summary of what was built, the model forward pass test results, and what you should study next before proceeding to Phase D.

---

## 🛠️ What Was Built in Phase C

1. **[`model/encoder.py`](file:///c:/Users/C12H28O9/OneDrive/Desktop/metal%20cutting%20optmising/model/encoder.py)**
   * **Transformer Multi-Head Self-Attention Encoder** ($d_{\text{model}}=128, L=2, \text{heads}=8$). Embeds piece dimensions into contextual vectors that capture global geometric relations across all items in the instance.
2. **[`model/decoder.py`](file:///c:/Users/C12H28O9/OneDrive/Desktop/metal%20cutting%20optmising/model/decoder.py)**
   * **Pointer Attention Decoder** (Vinyals et al. 2015, Kool et al. 2019). Computes attention compatibility logits between a graph query vector $\mathbf{q}$ and piece key vectors $\mathbf{K}$. Implements **Action Masking** (logits of placed pieces set to $-1\text{e}9$) and **Tanh Logit Clipping** ($C=10.0$).
3. **[`model/policy.py`](file:///c:/Users/C12H28O9/OneDrive/Desktop/metal%20cutting%20optmising/model/policy.py)**
   * The complete `AttentionPolicy` PyTorch module. Performs autoregressive rollout for $N$ steps, supports both `sample` (categorical) and `greedy` (argmax) decoding, and tracks trajectory log probabilities $\sum \log p_\theta(a_t | s_t)$.
4. **[`model/test_model.py`](file:///c:/Users/C12H28O9/OneDrive/Desktop/metal%20cutting%20optmising/model/test_model.py)**
   * Verifies end-to-end forward pass execution, output shapes, softmax probability normalization, action masking logic, and environment execution.
5. **[`model/README.md`](file:///c:/Users/C12H28O9/OneDrive/Desktop/metal%20cutting%20optmising/model/README.md)**
   * Plain-language explanation of Pointer Networks, Transformer Encoders, Action Masking, and Tanh Logit Clipping with literature citations.

---

## 📊 Numeric & Verification Summary

Running the Phase C model test script:

```text
=================================================================
  PHASE C: UNTRAINED POLICY NETWORK FORWARD PASS TEST
=================================================================
Batch Tensor Shape: [2, 10, 2] (batch_size=2, num_pieces=10, features=2)

[+] Forward Pass Execution Succeeded!
  Sampled Actions Shape   : [2, 10] -> [[8, 0, 1, 9, 6, 2, 7, 5, 3, 4], [8, 9, 7, 3, 0, 5, 4, 2, 1, 6]]
  Log Probs Sum (Sample) : [-14.95, -15.59]
  Greedy Actions Shape    : [2, 10] -> [[1, 9, 3, 7, 0, 8, 2, 4, 5, 6], [9, 5, 6, 7, 3, 1, 8, 2, 0, 4]]

VERIFYING PROBABILITY DISTRIBUTIONS & ACTION MASKING:
  Step 0 Probs Sum       : 1.000000 (Equal to 1.0)
  Step 0 Picked Piece    : Index 8
  Step 1 Prob of Piece 8: 0.000000 (Equal to 0.0 due to Action Masking!)
  [+] Action Masking Assertion PASSED!
-----------------------------------------------------------------
UNTRAINED POLICY LAYOUT SCORE:
  Sampled Ordering Execution Utilization: 61.41%
  (Untrained network weights yield random-like performance ~60%)
=================================================================
```

* **Softmax Normalization**: $\sum p_i = 1.000000$ verified.
* **Action Masking**: Probability of re-selecting piece 8 at step 1 is strictly $0.000000$ verified.
* **Git Commit**: Phase C committed to repository (`1e6ef93`).

---

## 📚 What You Should Go Study Next

Before we move on to **Phase D (REINFORCE Training Loop)**, review these Reinforcement Learning concepts:

1. **REINFORCE Algorithm (Policy Gradients - Williams, 1992)**:
   * Objective: Maximize expected reward $J(\theta) = \mathbb{E}_{\pi_\theta} [R(\tau)]$.
   * Policy Gradient Theorem:
     $$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=1}^N \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot \left( R(\tau) - b \right) \right]$$
   * Intuition: If a sampled piece sequence yields higher utilization than the baseline $b$ ($R(\tau) - b > 0$), we increase the probability of taking those actions. If lower ($R(\tau) - b < 0$), we decrease their probability.
2. **Moving-Average Baseline Variance Reduction**:
   * Baseline $b$: Tracked as an exponential moving average of average batch rewards:
     $$b_{\text{new}} = \beta \cdot b_{\text{old}} + (1 - \beta) \cdot \bar{R}_{\text{batch}} \quad (\text{where } \beta = 0.95)$$
   * Subtracting $b$ reduces gradient variance drastically without introducing bias into the gradient estimator.
3. **Loss Function in PyTorch**:
   * We minimize $\text{Loss} = - \text{mean} \left( \log p_\theta(\tau) \cdot (R(\tau) - b) \right)$.

