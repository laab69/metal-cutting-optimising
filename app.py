"""
Streamlit Web Application UI (`app.py`)

Interactive AI Sheet Metal Nesting Dashboard

Features:
1. Select Nesting Strategy: Trained AI Policy (Zero-Shot), Largest-First Heuristic, Random Policy, 90° Rotation, or Shapely Irregular Polygons.
2. Custom Instance Generation: Adjust number of pieces, sheet dimensions, random seed, or enter custom piece sizes.
3. Real-time Matplotlib Layout Rendering & Metrics (Utilization %, Latency ms, Placed Count).
4. Side-by-Side Comparison: Trained Neural Network vs. Classical Heuristic.
"""

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st

from env.generator import generate_instance
from env.nesting_env import NestingEnv
from baseline.largest_first import run_largest_first_heuristic
from model.policy import AttentionPolicy
from extensions.rotation_env import RotationNestingEnv
from extensions.rotation_policy import RotationAttentionPolicy
from extensions.polygon_env import PolygonNestingEnv, generate_polygon_instance


# Set Streamlit Page Config with wide layout and custom title
st.set_page_config(
    page_title="NCO Sheet Metal Nesting",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚙️ Neural Combinatorial Optimization for Sheet Metal Nesting")
st.caption("Deep Reinforcement Learning (Pointer Networks / Attention Model) for Zero-Shot Metal Part Nesting")


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.header("🎛️ Simulation Controls")

selected_mode = st.sidebar.selectbox(
    "Select Nesting Strategy",
    options=[
        "Trained Scaled AI Policy (N=20, Rollout Baseline)",
        "Trained Standard AI Policy (N=10)",
        "Largest-Piece-First Heuristic",
        "Random Placement Policy",
        "90° Rotation Action Space Policy",
        "Shapely Irregular Polygon Nesting"
    ]
)

st.sidebar.subheader("📐 Sheet & Piece Parameters")

sheet_w = st.sidebar.slider("Sheet Width", min_value=50.0, max_value=200.0, value=100.0, step=10.0)
sheet_h = st.sidebar.slider("Sheet Height", min_value=50.0, max_value=200.0, value=100.0, step=10.0)

num_pieces = st.sidebar.slider("Number of Pieces (N)", min_value=5, max_value=30, value=10 if "N=10" in selected_mode else 20, step=1)
rand_seed = st.sidebar.number_input("Random Seed", min_value=1, max_value=99999, value=42, step=1)

enable_side_by_side = st.sidebar.checkbox("Compare AI vs. Heuristic Side-by-Side", value=True)

# ---------------------------------------------------------
# Cached Model Loaders
# ---------------------------------------------------------
@st.cache_resource
def load_scaled_model(sheet_w, sheet_h):
    path = "model/scaled_policy.pt"
    policy = AttentionPolicy(input_dim=2, d_model=128, num_heads=8, num_layers=2, sheet_width=sheet_w, sheet_height=sheet_h)
    if os.path.exists(path):
        ckpt = torch.load(path, map_location='cpu')
        policy.load_state_dict(ckpt['model_state_dict'])
    policy.eval()
    return policy

@st.cache_resource
def load_standard_model(sheet_w, sheet_h):
    path = "model/trained_policy.pt"
    policy = AttentionPolicy(input_dim=2, d_model=128, num_heads=8, num_layers=2, sheet_width=sheet_w, sheet_height=sheet_h)
    if os.path.exists(path):
        ckpt = torch.load(path, map_location='cpu')
        policy.load_state_dict(ckpt['model_state_dict'])
    policy.eval()
    return policy


# Generate instance
instance_pieces = generate_instance(num_pieces=num_pieces, sheet_width=sheet_w, sheet_height=sheet_h, seed=rand_seed)


# ---------------------------------------------------------
# Helper Plotter Function
# ---------------------------------------------------------
def render_layout_plot(env, title_str, accent_color='darkblue'):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(patches.Rectangle((0, 0), env.sheet_width, env.sheet_height, linewidth=2, edgecolor='black', facecolor='#f8f9fa'))

    colors = plt.cm.tab20(np.linspace(0, 1, env.num_pieces))

    for idx, (x, y, w, h) in enumerate(env.placed_rects):
        orig_idx = env.placed_indices[idx]
        color = colors[orig_idx % len(colors)]

        rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor=accent_color, facecolor=color, alpha=0.85)
        ax.add_patch(rect)

        cx, cy = x + w / 2.0, y + h / 2.0
        ax.text(cx, cy, f"P{orig_idx}\n{w:.1f}x{h:.1f}", color='white', weight='bold', fontsize=8, ha='center', va='center')

    ax.set_xlim(-5, env.sheet_width + 5)
    ax.set_ylim(-5, env.sheet_height + 5)
    ax.set_aspect('equal')
    ax.set_title(title_str, fontweight='bold', fontsize=10)
    ax.set_xlabel("X (Sheet Width)")
    ax.set_ylabel("Y (Sheet Height)")
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------
# Execution & View Render
# ---------------------------------------------------------
env = NestingEnv(sheet_width=sheet_w, sheet_height=sheet_h, num_pieces=num_pieces)

t0 = time.time()

if selected_mode == "Trained Scaled AI Policy (N=20, Rollout Baseline)":
    policy = load_scaled_model(sheet_w, sheet_h)
    inst_tensor = torch.tensor(instance_pieces[np.newaxis, :, :], dtype=torch.float32)
    with torch.no_grad():
        actions, _, _ = policy(inst_tensor, decode_type="greedy")
        seq = actions[0].cpu().numpy()

    state = env.reset(pieces=instance_pieces)
    for act in seq:
        state, reward, done, _ = env.step(act)
    latency_ms = (time.time() - t0) * 1000.0
    utilization = env.score()
    mode_name = "Trained Scaled AI Policy"

elif selected_mode == "Trained Standard AI Policy (N=10)":
    policy = load_standard_model(sheet_w, sheet_h)
    inst_tensor = torch.tensor(instance_pieces[np.newaxis, :, :], dtype=torch.float32)
    with torch.no_grad():
        actions, _, _ = policy(inst_tensor, decode_type="greedy")
        seq = actions[0].cpu().numpy()

    state = env.reset(pieces=instance_pieces)
    for act in seq:
        state, reward, done, _ = env.step(act)
    latency_ms = (time.time() - t0) * 1000.0
    utilization = env.score()
    mode_name = "Trained Standard AI Policy"

elif selected_mode == "Largest-Piece-First Heuristic":
    utilization, _ = run_largest_first_heuristic(env, instance_pieces)
    latency_ms = (time.time() - t0) * 1000.0
    mode_name = "Largest-Piece-First Heuristic"

elif selected_mode == "Random Placement Policy":
    state = env.reset(pieces=instance_pieces)
    np.random.seed(rand_seed)
    done = False
    while not done:
        avail = np.where(state["mask"])[0]
        act = int(np.random.choice(avail))
        state, reward, done, _ = env.step(act)
    utilization = env.score()
    latency_ms = (time.time() - t0) * 1000.0
    mode_name = "Random Policy"

elif selected_mode == "90° Rotation Action Space Policy":
    rot_env = RotationNestingEnv(sheet_width=sheet_w, sheet_height=sheet_h, num_pieces=num_pieces)
    policy = RotationAttentionPolicy(input_dim=2, d_model=128, num_heads=8, num_layers=2, sheet_width=sheet_w, sheet_height=sheet_h)
    policy.eval()

    inst_tensor = torch.tensor(instance_pieces[np.newaxis, :, :], dtype=torch.float32)
    with torch.no_grad():
        actions, _ = policy(inst_tensor, decode_type="greedy")
        seq = actions[0].cpu().numpy()

    state = rot_env.reset(pieces=instance_pieces)
    for act in seq:
        state, reward, done, _ = rot_env.step(act)
    utilization = rot_env.score()
    latency_ms = (time.time() - t0) * 1000.0
    env = rot_env
    mode_name = "90° Rotation Policy"

elif selected_mode == "Shapely Irregular Polygon Nesting":
    poly_env = PolygonNestingEnv(sheet_width=sheet_w, sheet_height=sheet_h, num_pieces=num_pieces)
    polys = generate_polygon_instance(num_pieces=num_pieces, seed=rand_seed)

    areas = [p.area for p in polys]
    sorted_idx = np.argsort(-np.array(areas))

    state = poly_env.reset(polygons=polys)
    for act in sorted_idx:
        state, reward, done, _ = poly_env.step(act)

    utilization = poly_env.score()
    latency_ms = (time.time() - t0) * 1000.0

    # Custom Shapely Plot
    fig_poly, ax_p = plt.subplots(figsize=(6, 6))
    ax_p.add_patch(patches.Rectangle((0, 0), sheet_w, sheet_h, linewidth=2, edgecolor='black', facecolor='#f8f9fa'))
    colors = plt.cm.Set3(np.linspace(0, 1, num_pieces))

    for idx, poly in enumerate(poly_env.placed_polygons):
        orig_idx = poly_env.placed_indices[idx]
        color = colors[idx % len(colors)]
        x_coords, y_coords = poly.exterior.xy
        patch = patches.Polygon(list(zip(x_coords, y_coords)), closed=True, linewidth=1.5, edgecolor='darkgreen', facecolor=color, alpha=0.85)
        ax_p.add_patch(patch)
        ax_p.text(poly.centroid.x, poly.centroid.y, f"P{orig_idx}", color='black', weight='bold', fontsize=8, ha='center', va='center')

    ax_p.set_xlim(-5, sheet_w + 5)
    ax_p.set_ylim(-5, sheet_h + 5)
    ax_p.set_aspect('equal')
    ax_p.set_title(f"Shapely Irregular Polygons | Util: {utilization:.2f}%", fontweight='bold', fontsize=10)
    ax_p.grid(True, linestyle='--', alpha=0.5)

    mode_name = "Shapely Irregular Polygon Nesting"


# ---------------------------------------------------------
# Display Key Metrics Cards
# ---------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Sheet Utilization", f"{utilization:.2f}%")
col2.metric("Pieces Placed", f"{len(getattr(env, 'placed_rects', getattr(env, 'placed_polygons', [])))} / {num_pieces}")
col3.metric("Inference Latency", f"{latency_ms:.2f} ms")
col4.metric("Active Model Mode", mode_name)


st.divider()

# ---------------------------------------------------------
# Display Layout Visualizations
# ---------------------------------------------------------
if enable_side_by_side and selected_mode != "Shapely Irregular Polygon Nesting":
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader(f"🎯 Selected Strategy: {mode_name}")
        fig_selected = render_layout_plot(env, f"{mode_name}\nUtilization: {utilization:.2f}%")
        st.pyplot(fig_selected)

    with col_right:
        st.subheader("📏 Classical Heuristic (Largest-First)")
        env_h = NestingEnv(sheet_width=sheet_w, sheet_height=sheet_h, num_pieces=num_pieces)
        util_h, _ = run_largest_first_heuristic(env_h, instance_pieces)
        fig_h = render_layout_plot(env_h, f"Largest-Piece-First Heuristic\nUtilization: {util_h:.2f}%", accent_color='darkorange')
        st.pyplot(fig_h)

        diff = utilization - util_h
        if diff > 0:
            st.success(f"🚀 AI Strategy beats Largest-First Heuristic by **+{diff:.2f}%**!")
        elif diff < 0:
            st.warning(f"Largest-First Heuristic beats selected mode by **+{abs(diff):.2f}%**.")
        else:
            st.info("Both strategies produced identical utilization scores.")
else:
    st.subheader(f"Layout View: {mode_name}")
    if selected_mode == "Shapely Irregular Polygon Nesting":
        st.pyplot(fig_poly)
    else:
        fig_single = render_layout_plot(env, f"{mode_name}\nUtilization: {utilization:.2f}%")
        st.pyplot(fig_single)

st.divider()

# ---------------------------------------------------------
# Piece Dimension Table
# ---------------------------------------------------------
with st.expander("📊 View Generated Piece Dimensions Table"):
    if selected_mode != "Shapely Irregular Polygon Nesting":
        st.dataframe(
            [{"Piece Index": i, "Width": f"{w:.2f}", "Height": f"{h:.2f}", "Area": f"{w*h:.2f}"} for i, (w, h) in enumerate(instance_pieces)],
            use_container_width=True
        )
    else:
        st.write("Irregular polygon templates generated: L-Shapes, Triangles, T-Shapes, Trapezoids, Rectangles.")
