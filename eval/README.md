# Generalization & Zero-Shot Evaluation (`eval/`)

Welcome to **Phase E** of the Neural Combinatorial Optimization (NCO) project!

---

## 💡 What is Zero-Shot Generalization in NCO?

This phase represents **the main objective of the entire project**: demonstrating that a neural network trained via Reinforcement Learning can generalize to **new, unseen problem instances** without re-training or re-solving from scratch.

### Learned Policy vs. Classical Metaheuristics & Rules

| Dimension | Classical Metaheuristics (GA / SA) | Classical Heuristics (Largest-First) | Neural Policy (Our Model) |
| :--- | :--- | :--- | :--- |
| **How it works** | Restarts optimization search from scratch for *every* new instance | Applies a hardcoded static rule (descending area) | Single forward pass through trained Transformer network |
| **Time per instance** | Slow (seconds to minutes) | Fast ($O(N \log N)$) | Extremely Fast ($O(N)$ milliseconds) |
| **Adaptability** | Re-solves from scratch | Fixed rule (cannot adapt to shape distribution) | Learns complex spatial pattern relationships |
| **Generalization** | No cross-instance learning | Fixed rule | Zero-Shot generalization on unseen test sets |

---

## 📐 Evaluation Methodology

1. **Held-Out Test Set**: We generate 200 completely new, unseen problem instances using a dedicated test seed (`seed=5555`). The policy network was never exposed to these instances during training.
2. **Zero-Shot Inference**: We load the saved model checkpoint (`model/trained_policy.pt`) and run a single greedy forward pass per instance (`decode_type='greedy'`).
3. **Benchmarked Methods**:
   * **Random Policy**: Baseline lower bound (Phase A).
   * **Largest-Piece-First Heuristic**: Classical benchmark floor (Phase B).
   * **Untrained Policy Network**: Model with initial random weights (Phase C).
   * **Trained Attention Policy Network**: Our RL-trained Transformer model (Phase D).
4. **Metrics Tracked**:
   * **Mean Utilization (%)**: $\frac{1}{M} \sum_{i=1}^M \text{Utilization}_i$
   * **Standard Deviation (%)**: Measure of consistency across problem instances.
   * **Wall-Clock Time per Instance (ms)**: Computation latency per problem instance.

---

## 📁 File Structure

* [`heldout_generator.py`](file:///c:/Users/C12H28O9/OneDrive/Desktop/metal%20cutting%20optmising/eval/heldout_generator.py): Generates reproducible held-out test datasets.
* [`evaluate_generalization.py`](file:///c:/Users/C12H28O9/OneDrive/Desktop/metal%20cutting%20optmising/eval/evaluate_generalization.py): Loads trained weights, runs zero-shot inference, benchmarks against all baselines, and generates `eval_generalization_comparison.png`.

---

## 🚀 How to Run the Generalization Test

```bash
python -m eval.evaluate_generalization
```
This script evaluates 200 held-out instances, outputs performance statistics to stdout, and exports a visual plot to `eval_generalization_comparison.png`.
