# Environment & Nesting Plumbing (`env/`)

Welcome to **Phase A** of the Neural Combinatorial Optimization (NCO) for Sheet Metal Nesting project!

---

## 💡 Core RL Concepts Explained

In Reinforcement Learning (RL), an **Agent** learns to make decisions by interacting with an **Environment**. In our project, the goal is to pack a set of rectangular metal pieces onto a fixed-size sheet to maximize space utilization.

Here is how standard RL terminology maps directly to our sheet nesting problem:

### 1. State ($s_t$)
* **What it means in RL:** The current snapshot of the world that the decision-maker sees.
* **In our environment:** At step $t$, the state consists of:
  1. The collection of **remaining unplaced pieces** (each represented by its `[width, height]`).
  2. The current occupancy grid / layout of the sheet (where pieces have already been placed).

### 2. Action ($a_t$)
* **What it means in RL:** The choice selected by the policy at step $t$.
* **In our environment:** Selecting **which** remaining piece to place next (an index referencing a piece in the remaining pool). This matches the **Pointer Network** formulation (Vinyals et al., 2015; Kool et al., 2019), where the output is a discrete selection over a variable number of remaining candidate items.

### 3. Placement Decoder (Deterministic Plumbing)
* **What it is:** A non-learned, rule-based heuristic function (e.g., Bottom-Left / Lowest-then-Leftmost).
* **Why we need it:** The neural policy's job is **sequencing** (deciding the *order* in which pieces are handed to the machine). Once the policy picks a piece, the placement decoder determines its exact $(x, y)$ coordinate on the sheet. 
* **Important Note:** This decoder is purely fixed infrastructure (< 30 lines of code) so that an episode can yield a physical layout to score. We are **not** optimizing or tuning this decoder; all learning will happen in the neural network's sequencing policy in later phases.

### 4. Reward ($R$)
* **What it means in RL:** The scalar feedback signal evaluating performance.
* **In our environment:** 
  * Intermediate steps ($t < N$): $R_t = 0$ (Delayed reward).
  * Final step ($t = N$): $R_{\text{final}} = \frac{\sum \text{Placed Piece Areas}}{\text{Total Sheet Area}}$.
* **Why sparse rewards?** In combinatorial optimization (like Traveling Salesperson or Nesting), partial layouts don't accurately reflect final quality. A sequence that looks good early on might leave awkward gaps at the end. Thus, the agent receives its reward only when the full episode finishes.

---

## 📁 File Structure

* [`generator.py`](file:///c:/Users/C12H28O9/OneDrive/Desktop/metal%20cutting%20optmising/env/generator.py): Generates random problem instances (sets of $N$ rectangular pieces with fixed sheet dimensions).
* [`decoder.py`](file:///c:/Users/C12H28O9/OneDrive/Desktop/metal%20cutting%20optmising/env/decoder.py): The deterministic Bottom-Left placement algorithm (< 30 lines).
* [`nesting_env.py`](file:///c:/Users/C12H28O9/OneDrive/Desktop/metal%20cutting%20optmising/env/nesting_env.py): The main Environment class defining `reset()`, `step()`, and `score()`.
* [`sanity_check.py`](file:///c:/Users/C12H28O9/OneDrive/Desktop/metal%20cutting%20optmising/env/sanity_check.py): Runs a random policy through the environment, prints metrics, and displays/saves visual plots.

---

## 🚀 How to Run the Sanity Check

```bash
python -m env.sanity_check
```
This will execute a random placement sequence, output utilization metrics, and save a layout plot to `env_sanity_check.png`.
