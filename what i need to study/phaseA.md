

# Phase A Complete! 🎉

Here is a summary of what was built, the output produced, and what you should study next before proceeding to Phase B.

---

## 🛠️ What Was Built in Phase A

1. **[`env/generator.py`](file:///c:/Users/C12H28O9/OneDrive/Desktop/metal%20cutting%20optmising/env/generator.py)**
   * Generates random problem instances on the fly ($N=10$ rectangular pieces with widths and heights uniformly sampled between 10.0 and 40.0 on a 100x100 sheet).
2. **[`env/decoder.py`](file:///c:/Users/C12H28O9/OneDrive/Desktop/metal%20cutting%20optmising/env/decoder.py)**
   * The non-learned placement decoder (<30 lines). Uses a deterministic **Bottom-Left** (Lowest-then-Leftmost) coordinate search to fit selected pieces on the sheet without overlap.
3. **[`env/nesting_env.py`](file:///c:/Users/C12H28O9/OneDrive/Desktop/metal%20cutting%20optmising/env/nesting_env.py)**
   * The RL environment encapsulating State $s_t$, Action masking (ensuring unplaced pieces are available and placed pieces cannot be re-selected), `step()`, `reset()`, and delayed scalar Reward $R$ (final utilization %).
4. **[`env/sanity_check.py`](file:///c:/Users/C12H28O9/OneDrive/Desktop/metal%20cutting%20optmising/env/sanity_check.py)**
   * Runs a **Random Policy** sanity test, prints episode step-by-step logs, and exports a visual layout plot to [`env_sanity_check.png`](file:///c:/Users/C12H28O9/OneDrive/Desktop/metal%20cutting%20optmising/env_sanity_check.png).
5. **[`env/README.md`](file:///c:/Users/C12H28O9/OneDrive/Desktop/metal%20cutting%20optmising/env/README.md)**
   * Beginner-friendly introduction explaining state, action, action masking, placement decoder, and sparse rewards using paper vocabulary.

---

## 📊 Numeric & Visual Verification

Running the Phase A sanity script on a sample 10-piece instance:

```text
============================================================
  PHASE A: ENVIRONMENT & PLUMBING SANITY RUN (RANDOM POLICY)
============================================================
Total Pieces Placed: 10 / 10
Final Utilization  : 61.41%
Layout plot saved  : env_sanity_check.png
```

* **Visual Output**: The generated layout image [`env_sanity_check.png`](file:///c:/Users/C12H28O9/OneDrive/Desktop/metal%20cutting%20optmising/env_sanity_check.png) shows all 10 pieces tightly packed into the bottom-left corner of the 100x100 sheet without overlaps.
* **Git Commit**: Phase A changes committed to repository (`d1136ef`).

---

## 📚 What You Should Go Study Next

Before we move on to **Phase B (Baseline Heuristic)**, spend a few minutes reviewing:

1. **Greedy Packing Heuristics (Largest First)**:
   * Research why classical algorithms like *First-Fit Decreasing (FFD)* or *Largest-Area-First* are effective baseline heuristics for 1D/2D Bin Packing.
   * Notice how sorting pieces by area (or max dimension) before feeding them to a placement decoder consistently packs larger shapes first, preventing small pieces from creating unusable gaps early on.
2. **The Concept of a Baseline Floor**:
   * In Neural Combinatorial Optimization papers (e.g., Kool et al. 2019, Bello et al. 2016), machine learning models are evaluated against deterministic heuristics. A neural network is only useful if it learns a policy that beats simple classical rules like "largest-piece-first".

