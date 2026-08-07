# Baseline Heuristics (`baseline/`)

Welcome to **Phase B** of the Neural Combinatorial Optimization (NCO) project!

---

## 💡 What is a Baseline Floor Heuristic?

In Machine Learning and Reinforcement Learning, a **Baseline Floor** is a benchmark score achieved by a simple, non-learned, deterministic heuristic rule (e.g., sorting items by size before packing them).

### Why do we need a Baseline Floor?
1. **Sanity Check**: Before declaring that an RL model has "learned" something useful, we must verify that its performance exceeds trivial rule-of-thumb heuristics. If a complex Transformer model gets 65% utilization, but sorting by size gets 75%, the neural model hasn't actually learned anything useful!
2. **Minimum Performance Target**: In papers like Kool et al. (2019) and Bello et al. (2016), classical heuristics (like *Farthest Insertion* for TSP or *First-Fit Decreasing* for Bin Packing) serve as the benchmark number that the RL agent must beat.

---

## 📐 The Heuristic: Largest-Piece-First (Largest Area First)

In 2D sheet metal nesting and rectangle packing:
* **The Rule**: Calculate the area ($W \times H$) of each piece in the instance. Sort the remaining unplaced pieces in **descending order of area** (largest pieces first), and feed them one by one to the same deterministic placement decoder ([`place_bottom_left`](file:///c:/Users/C12H28O9/OneDrive/Desktop/metal%20cutting%20optmising/env/decoder.py)).
* **Why Largest-First Works**: Larger pieces are harder to fit into remaining tight gaps later in the episode. By placing large pieces early while the sheet is completely empty, small pieces can easily fill in the remaining small gaps later.

---

## 📁 File Structure

* [`largest_first.py`](file:///c:/Users/C12H28O9/OneDrive/Desktop/metal%20cutting%20optmising/baseline/largest_first.py): Implements the Largest-Piece-First ordering heuristic.
* [`evaluate_baseline.py`](file:///c:/Users/C12H28O9/OneDrive/Desktop/metal%20cutting%20optmising/baseline/evaluate_baseline.py): Evaluates both the Random Policy and Largest-Piece-First Heuristic across 100 random instances, reporting average utilization %, standard deviation, and saving a comparison plot.

---

## 🚀 How to Run the Evaluation

```bash
python -m baseline.evaluate_baseline
```
This script will evaluate 100 random instances and output the average utilization % and standard deviation, saving a bar chart plot to `baseline_comparison.png`.
