
# Phase B Complete! 🎉

Here is a summary of what was built, the benchmark floor established, and what you should study next before proceeding to Phase C.

---

## 🛠️ What Was Built in Phase B

1. **[`baseline/largest_first.py`](file:///c:/Users/C12H28O9/OneDrive/Desktop/metal%20cutting%20optmising/baseline/largest_first.py)**
   * Implements the **Largest-Piece-First** (Largest Area First) deterministic heuristic rule. Sorts remaining unplaced pieces by area ($W \times H$) in descending order before sending them to the placement decoder.
2. **[`baseline/evaluate_baseline.py`](file:///c:/Users/C12H28O9/OneDrive/Desktop/metal%20cutting%20optmising/baseline/evaluate_baseline.py)**
   * Benchmarks both the Random Policy and Largest-Piece-First heuristic across **100 randomly generated problem instances** to calculate statistically reliable performance metrics.
   * Generates and saves a comparison plot to [`baseline_comparison.png`](file:///c:/Users/C12H28O9/OneDrive/Desktop/metal%20cutting%20optmising/baseline_comparison.png).
3. **[`baseline/README.md`](file:///c:/Users/C12H28O9/OneDrive/Desktop/metal%20cutting%20optmising/baseline/README.md)**
   * Beginner-friendly introduction explaining why baseline benchmark floors are mandatory in AI/RL projects before building neural network models.

---

## 📊 Numeric & Visual Verification

Running the 100-instance evaluation script yielded:

```text
=================================================================
  PHASE B: EVALUATING BASELINE HEURISTIC OVER 100 INSTANCES
=================================================================
BENCHMARK RESULTS SUMMARY:
-----------------------------------------------------------------
  Total Evaluation Instances : 100
  Pieces Per Instance        : 10
  Sheet Dimensions           : 100.0 x 100.0
  Total Wall-Clock Time      : 0.091 seconds (0.91 ms/instance)
-----------------------------------------------------------------
  1. Random Policy           : 60.27% +/- 7.19%
  2. Largest-Piece-First     : 62.62% +/- 8.60%
-----------------------------------------------------------------
  [+] Baseline Benchmark Floor  : 62.62% utilization
  [+] Heuristic Gain over Random: +2.35% percentage points
=================================================================
```

* **Target Benchmark Floor**: **`62.62%`** average sheet utilization.
* **Key Takeaway**: Any neural network policy we build in Phase C & D **must exceed 62.62% average utilization** to prove that it has learned a non-trivial spatial policy superior to standard greedy rules.
* **Visual Output**: Plot saved to [`baseline_comparison.png`](file:///c:/Users/C12H28O9/OneDrive/Desktop/metal%20cutting%20optmising/baseline_comparison.png).
* **Git Commit**: Phase B committed to repository (`e677018`).

---

## 📚 What You Should Go Study Next

Before we move on to **Phase C (Policy Network - Encoder & Pointer Decoder)**, review these core neural network concepts:

1. **Pointer Networks (Vinyals et al. 2015)**:
   * Learn why standard classification outputs (fixed vocabulary size) cannot solve combinatorial optimization problems where the output sequence consists of elements from the input set.
   * Understand how attention scores act as "pointers" over variable-length inputs.
2. **Attention Model Architecture (Kool et al. 2019)**:
   * **Encoder**: Uses multi-head self-attention (MHSA) to transform raw piece features `[width, height]` into rich contextual embedding vectors ($d_{\text{model}} = 128$).
   * **Decoder**: Autoregressively picks piece indices step-by-step. Uses a query vector (containing graph/sheet context) to compute attention logits against remaining piece embeddings.
3. **Action Masking**:
   * Learn how setting the attention logits of already-placed pieces to $-\infty$ (or `-1e9`) enforces that $\text{softmax}(-\infty) = 0$, guaranteeing the policy never samples an already-placed piece.

