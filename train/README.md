# REINFORCE Training Loop (`train/`)

Welcome to **Phase D** of the Neural Combinatorial Optimization (NCO) project!

---

## 💡 Policy Gradients & REINFORCE Algorithm

In Supervised Learning, neural networks are trained with ground-truth targets (e.g., "this image is a dog"). In Combinatorial Optimization (like Sheet Metal Nesting), we do **not** know the optimal piece ordering in advance!

Instead, we use **Reinforcement Learning (REINFORCE - Williams, 1992)** to learn directly from trial-and-error rewards.

```
┌───────────────────────────┐
│ Batch of Sheet Instances  │
└───────────────────────────┘
              │
              ▼
┌───────────────────────────┐
│ Attention Policy Network  │ ──► Sample piece sequences τ_i ~ π_θ
└───────────────────────────┘
              │
              ▼
┌───────────────────────────┐
│ Nesting Placement Decoder │ ──► Execute sequences in NestingEnv
└───────────────────────────┘ ──► Get scalar rewards R(τ_i) (utilization %)
              │
              ▼
┌───────────────────────────┐
│ REINFORCE Policy Gradient │ ──► Advantage A_i = R(τ_i) - Baseline b
└───────────────────────────┘ ──► Loss = - mean( log p_θ(τ_i) * A_i )
              │
              ▼
┌───────────────────────────┐
│ PyTorch Adam Optimizer    │ ──► Update weights θ
└───────────────────────────┘
```

---

## 📐 Mathematical Derivation

### 1. Objective Function
We aim to maximize the expected final utilization reward $J(\theta)$:
$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]$$

### 2. Policy Gradient Theorem
Using the log-derivative trick ($\nabla f(x) = f(x) \nabla \log f(x)$):
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log p_\theta(\tau) \cdot (R(\tau) - b) \right]$$
where $\log p_\theta(\tau) = \sum_{t=1}^N \log \pi_\theta(a_t | s_t)$ is the log probability of the full trajectory.

### 3. Variance Reduction with Exponential Moving Average (EMA) Baseline
* Without a baseline ($b=0$), policy gradient updates suffer from extremely high variance because rewards $R(\tau) \in [50\%, 80\%]$ are always positive.
* We track an **Exponential Moving Average Baseline ($b$)**:
  $$b \leftarrow \beta \cdot b + (1 - \beta) \cdot \bar{R}_{\text{batch}} \quad (\text{where } \beta = 0.95)$$
* **Advantage Signal ($A_i = R_i - b$)**:
  * If $A_i > 0$: The sampled sequence performed **better** than recent average $\Rightarrow$ Increase action probabilities!
  * If $A_i < 0$: The sampled sequence performed **worse** than recent average $\Rightarrow$ Decrease action probabilities!

### 4. PyTorch Policy Loss
Because PyTorch performs gradient *descent* (minimizing loss), we negate the policy gradient objective:
$$\mathcal{L}(\theta) = - \frac{1}{B} \sum_{i=1}^B \left( \log p_\theta(\tau_i) \cdot (R(\tau_i) - b) \right)$$

---

## 📁 File Structure

* [`moving_average.py`](file:///c:/Users/C12H28O9/OneDrive/Desktop/metal%20cutting%20optmising/train/moving_average.py): Exponential Moving Average (EMA) baseline tracker.
* [`trainer.py`](file:///c:/Users/C12H28O9/OneDrive/Desktop/metal%20cutting%20optmising/train/trainer.py): Core REINFORCE training loop, gradient clipping, metrics logging, and checkpointing.
* [`train_policy.py`](file:///c:/Users/C12H28O9/OneDrive/Desktop/metal%20cutting%20optmising/train/train_policy.py): Executable script to start training and save convergence plots to `training_curve.png`.

---

## 🚀 How to Run Training

```bash
python -m train.train_policy
```
This will train the policy over 400 steps, log metrics to console, save the trained model weights to `model/trained_policy.pt`, and generate `training_curve.png`.
