# Attention Policy Network (`model/`)

Welcome to **Phase C** of the Neural Combinatorial Optimization (NCO) project!

---

## 💡 Neural Architecture Overview (Kool et al., 2019 & Vinyals et al., 2015)

In classical deep learning, a classifier outputs a probability distribution over a fixed set of classes (e.g., 1000 ImageNet categories). However, in sheet metal nesting, the set of available pieces changes dynamically as pieces are placed.

To solve this, we implement the **Attention Model** architecture (Kool, van Hoof, & Welling, 2019), which builds on **Pointer Networks** (Vinyals, Fortunato, & Jaitly, 2015).

```
   Raw Pieces [N x 2] (w, h)
           │
           ▼
┌─────────────────────────┐
│   Linear Feature Embed  │ ──► (N x d_model)
└─────────────────────────┘
           │
           ▼
┌─────────────────────────┐
│ Transformer Encoder     │ ──► Contextual Piece Embeddings H (N x d_model)
│ (Multi-Head Self-Attn)  │     + Graph Mean Embedding h_bar (1 x d_model)
└─────────────────────────┘
           │
           ▼
┌─────────────────────────┐
│ Pointer Decoder         │ ──► Compute Query q from graph context & last item
│ (Pointer Attention)     │ ──► Compute Logits u_i = C * tanh( (q W_Q)(h_i W_K)^T / sqrt(d_k) )
└─────────────────────────┘
           │
           ▼
┌─────────────────────────┐
│ Action Masking & Softmax│ ──► Set placed piece logits to -inf
└─────────────────────────┘ ──► Output probability distribution P (1 x N)
```

---

## 🔑 Key Architectural Components

### 1. Encoder (Transformer Self-Attention)
* **Input**: $N$ rectangular pieces, each described by normalized dimensions $[w_i / W_{\text{sheet}}, h_i / H_{\text{sheet}}]$.
* **Linear Projection**: Embeds 2D piece dimensions into a $d_{\text{model}}$-dimensional vector space ($d_{\text{model}} = 128$).
* **Self-Attention Layers**: $L=2$ Multi-Head Self-Attention (MHSA) blocks allow pieces to "talk" to each other and gain contextual awareness of the overall instance shape distribution.

### 2. Decoder (Pointer Mechanism)
* **Query Vector ($\mathbf{q}$)**: Synthesizes graph-level context ($\mathbf{\bar{h}} = \text{mean}(\mathbf{H})$) and state context (the embedding of the previously placed piece).
* **Pointer Logits ($\mathbf{u}$)**: Computes dot-product attention scores between query $\mathbf{q}$ and piece embeddings $\mathbf{h}_i$:
  $$u_i = C \cdot \tanh\left( \frac{\mathbf{q} W_Q (\mathbf{h}_i W_K)^T}{\sqrt{d_k}} \right)$$
  *(where $C = 10.0$ is the logit clipping constant from Kool et al. to stabilize training).*

### 3. Action Masking
* **Why it is mandatory**: A piece can only be cut once. If piece $i$ is already placed, its mask value is `False`.
* **Implementation**: We set $u_i = -\infty$ for masked pieces. Since $\exp(-\infty) = 0$, the softmax probability $\sigma(u)_i$ is strictly $0.0$, guaranteeing the policy never selects an already-placed piece.

### 4. Categorical Sampling vs. Greedy Decoding
* **Sampling Mode**: $a_t \sim \text{Categorical}(\mathbf{p})$. Used during REINFORCE training to explore different sequence ordering decisions.
* **Greedy Mode**: $a_t = \arg\max(\mathbf{p})$. Picks the highest probability piece at each step. Used during inference/testing.

---

## 📁 File Structure

* [`encoder.py`](file:///c:/Users/C12H28O9/OneDrive/Desktop/metal%20cutting%20optmising/model/encoder.py): Transformer Multi-Head Self-Attention Encoder.
* [`decoder.py`](file:///c:/Users/C12H28O9/OneDrive/Desktop/metal%20cutting%20optmising/model/decoder.py): Pointer Decoder with action masking and logit clipping.
* [`policy.py`](file:///c:/Users/C12H28O9/OneDrive/Desktop/metal%20cutting%20optmising/model/policy.py): `AttentionPolicy` class combining Encoder + Decoder into an end-to-end PyTorch module.
* [`test_model.py`](file:///c:/Users/C12H28O9/OneDrive/Desktop/metal%20cutting%20optmising/model/test_model.py): Forward pass test verifying probability distributions, action sampling, log probability tracking, and environment execution.

---

## 🚀 How to Run the Model Forward Pass Test

```bash
python -m model.test_model
```
This script runs a full forward pass through the untrained PyTorch network, verifies shape dimensions and probability distributions, and displays step-by-step sampling logits.
