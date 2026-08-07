"""
Industrial Sheet Metal Nesting AI — Final Production App (`app.py`)

A production-ready UI for factory operators and engineers:
1. Input Sheet Metal Stock Dimensions (Width, Height).
2. Input Custom Parts Inventory (Add Rectangles, Rotated Parts, L-Shapes, Triangles, T-Shapes, Trapezoids).
3. Execute Unified AI Nesting Engine (90° Rotation Policy + Shapely 2D Geometry Placement).
4. View High-Resolution Nested Sheet Layout, Utilization %, Scrap Ratio, and Cutting Manifest Table.
"""

import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st
import shapely
from shapely.geometry import Polygon, box
from shapely.affinity import translate

from extensions.rotation_policy import RotationAttentionPolicy


# ---------------------------------------------------------
# Page Configuration & Custom CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Industrial Metal Nesting AI",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 0px; }
    .sub-header { font-size: 1.0rem; color: #64748B; margin-bottom: 20px; }
    .stButton>button { width: 100%; font-weight: bold; background-color: #0F172A; color: white; border-radius: 8px; height: 48px; }
    .metric-card { background-color: #F8FAFC; padding: 15px; border-radius: 8px; border: 1px solid #E2E8F0; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🔩 Industrial Sheet Metal Nesting AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Production Nesting Engine: Transformer Attention Policy + 90° Rotation + Shapely 2D Geometry</div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# Shapely Polygon Templates
# ---------------------------------------------------------
SHAPE_LIBRARY = {
    "Rectangle": lambda w, h: Polygon([(0, 0), (w, 0), (w, h), (0, h)]),
    "L-Shape": lambda w, h: Polygon([(0, 0), (w, 0), (w, h*0.4), (w*0.4, h*0.4), (w*0.4, h), (0, h)]),
    "Triangle": lambda w, h: Polygon([(0, 0), (w, 0), (w*0.5, h)]),
    "T-Shape": lambda w, h: Polygon([(0, h*0.6), (w, h*0.6), (w, h), (w*0.65, h), (w*0.65, 0), (w*0.35, 0), (w*0.35, h), (0, h)]),
    "Trapezoid": lambda w, h: Polygon([(0, 0), (w, 0), (w*0.7, h), (w*0.3, h)])
}


# ---------------------------------------------------------
# Sidebar: Stock Sheet Configuration
# ---------------------------------------------------------
st.sidebar.header("📐 1. Metal Sheet Stock Size")
sheet_width = st.sidebar.number_input("Sheet Width (mm)", min_value=50.0, max_value=500.0, value=100.0, step=10.0)
sheet_height = st.sidebar.number_input("Sheet Height (mm)", min_value=50.0, max_value=500.0, value=100.0, step=10.0)

st.sidebar.divider()
st.sidebar.header("⚙️ 2. Nesting Options")
allow_rotation = st.sidebar.checkbox("Allow 90° Piece Rotation", value=True)
grid_granularity = st.sidebar.slider("Placement Grid Step (mm)", min_value=1.0, max_value=5.0, value=2.0, step=0.5)


# ---------------------------------------------------------
# Main UI: Parts Inventory Input
# ---------------------------------------------------------
st.subheader("📦 2. Parts Inventory to Cut")

col_left, col_right = st.columns([2, 1])

with col_right:
    st.markdown("### 🎲 Preset Inventory")
    if st.button("Generate Sample Factory Order"):
        st.session_state["parts_df"] = pd.DataFrame([
            {"Part ID": "P1", "Shape": "Rectangle", "Width (mm)": 38.5, "Height (mm)": 39.1, "Quantity": 1},
            {"Part ID": "P2", "Shape": "L-Shape", "Width (mm)": 32.0, "Height (mm)": 35.0, "Quantity": 1},
            {"Part ID": "P3", "Shape": "Triangle", "Width (mm)": 36.0, "Height (mm)": 25.7, "Quantity": 1},
            {"Part ID": "P4", "Shape": "Rectangle", "Width (mm)": 28.0, "Height (mm)": 23.0, "Quantity": 2},
            {"Part ID": "P5", "Shape": "T-Shape", "Width (mm)": 31.2, "Height (mm)": 28.7, "Quantity": 1},
            {"Part ID": "P6", "Shape": "Trapezoid", "Width (mm)": 25.0, "Height (mm)": 20.0, "Quantity": 1},
            {"Part ID": "P7", "Shape": "Rectangle", "Width (mm)": 21.2, "Height (mm)": 15.6, "Quantity": 2},
        ])

if "parts_df" not in st.session_state:
    st.session_state["parts_df"] = pd.DataFrame([
        {"Part ID": "Part_1", "Shape": "Rectangle", "Width (mm)": 35.0, "Height (mm)": 25.0, "Quantity": 2},
        {"Part ID": "Part_2", "Shape": "L-Shape", "Width (mm)": 30.0, "Height (mm)": 30.0, "Quantity": 2},
        {"Part ID": "Part_3", "Shape": "Triangle", "Width (mm)": 28.0, "Height (mm)": 20.0, "Quantity": 2},
        {"Part ID": "Part_4", "Shape": "Rectangle", "Width (mm)": 20.0, "Height (mm)": 15.0, "Quantity": 3},
    ])

with col_left:
    st.markdown("### 📝 Edit Part Order List")
    edited_df = st.data_editor(
        st.session_state["parts_df"],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Shape": st.column_config.SelectboxColumn("Shape Type", options=list(SHAPE_LIBRARY.keys()), required=True),
            "Width (mm)": st.column_config.NumberColumn("Width (mm)", min_value=5.0, max_value=200.0, default=20.0),
            "Height (mm)": st.column_config.NumberColumn("Height (mm)", min_value=5.0, max_value=200.0, default=20.0),
            "Quantity": st.column_config.NumberColumn("Quantity", min_value=1, max_value=10, default=1)
        }
    )

st.divider()

# ---------------------------------------------------------
# Run AI Nesting Engine Button
# ---------------------------------------------------------
if st.button("⚡ EXECUTE AI NESTING ENGINE"):
    st.markdown("---")
    
    # 1. Expand dataframe rows based on Quantity into individual Shapely Polygon items
    expanded_items = []
    for idx, row in edited_df.iterrows():
        name = str(row["Part ID"])
        shape_type = str(row["Shape"])
        w = float(row["Width (mm)"])
        h = float(row["Height (mm)"])
        qty = int(row["Quantity"])

        polygon_fn = SHAPE_LIBRARY.get(shape_type, SHAPE_LIBRARY["Rectangle"])

        for q in range(qty):
            poly = polygon_fn(w, h)
            expanded_items.append({
                "id": f"{name}_{q+1}" if qty > 1 else name,
                "shape": shape_type,
                "width": w,
                "height": h,
                "polygon": poly,
                "area": poly.area
            })

    num_items = len(expanded_items)
    if num_items == 0:
        st.error("Please add at least 1 part to the order list!")
        st.stop()

    t_start = time.time()

    # 2. AI Attention Policy Neural Ordering
    # Pass piece bounding boxes [w, h] to Rotation-Aware Attention Policy
    piece_features_np = np.array([[item["width"], item["height"]] for item in expanded_items], dtype=np.float32)
    
    policy = RotationAttentionPolicy(
        input_dim=2,
        d_model=128,
        num_heads=8,
        num_layers=2,
        sheet_width=sheet_width,
        sheet_height=sheet_height
    )
    # Load trained policy checkpoint if available
    ckpt_path = "model/scaled_policy.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        # Load matched encoder/decoder weights safely
        policy.load_state_dict(ckpt['model_state_dict'], strict=False)
    policy.eval()

    # Neural Forward Pass in greedy mode
    batch_t = torch.tensor(piece_features_np[np.newaxis, :, :], dtype=torch.float32)
    with torch.no_grad():
        actions_t, _ = policy(batch_t, decode_type="greedy")
        ai_action_sequence = actions_t[0].cpu().numpy()

    # 3. Geometry Placement Engine (Shapely + Bottom-Left Search)
    sheet_poly = box(0, 0, sheet_width, sheet_height)
    sheet_area = sheet_width * sheet_height

    placed_polygons = []
    placement_manifest = []
    used_mask = np.ones(2 * num_items, dtype=bool)

    for action_val in ai_action_sequence:
        piece_idx = int(action_val % num_items)
        is_rotated = bool(action_val >= num_items) if allow_rotation else False

        if not used_mask[piece_idx]:
            continue  # Already placed

        item = expanded_items[piece_idx]
        poly = item["polygon"]

        # Rotate polygon 90° if selected by policy
        if is_rotated:
            poly = shapely.affinity.rotate(poly, 90, origin=(0, 0))

        # Search bottom-left position on sheet grid
        minx, miny, maxx, maxy = poly.bounds
        p_w, p_h = maxx - minx, maxy - miny

        xs = np.arange(0.0, sheet_width - p_w + 0.5, grid_granularity)
        ys = np.arange(0.0, sheet_height - p_h + 0.5, grid_granularity)

        placed_loc = None
        for y in ys:
            for x in xs:
                shifted = translate(poly, xoff=x - minx, yoff=y - miny)
                if not sheet_poly.contains(shifted):
                    continue

                overlap = any(shifted.intersects(p) and not shifted.touches(p) for p in placed_polygons)
                if not overlap:
                    placed_loc = (x, y, shifted)
                    break
            if placed_loc is not None:
                break

        # Update placement state
        used_mask[piece_idx] = False
        used_mask[piece_idx + num_items] = False

        if placed_loc is not None:
            px, py, final_poly = placed_loc
            placed_polygons.append(final_poly)
            placement_manifest.append({
                "Part ID": item["id"],
                "Shape": item["shape"],
                "Placed": "YES",
                "Position (X, Y)": f"({px:.1f}, {py:.1f})",
                "Orientation": "90° Rotated" if is_rotated else "0° Normal",
                "Area (mm²)": f"{item['area']:.1f}"
            })
        else:
            placement_manifest.append({
                "Part ID": item["id"],
                "Shape": item["shape"],
                "Placed": "NO (Sheet Full)",
                "Position (X, Y)": "-",
                "Orientation": "-",
                "Area (mm²)": f"{item['area']:.1f}"
            })

    nest_latency_ms = (time.time() - t_start) * 1000.0

    total_placed_area = sum(p.area for p in placed_polygons)
    utilization_pct = (total_placed_area / sheet_area) * 100.0
    scrap_pct = 100.0 - utilization_pct
    placed_count = len(placed_polygons)

    # 4. Display Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sheet Utilization", f"{utilization_pct:.2f}%")
    m2.metric("Scrap Material Ratio", f"{scrap_pct:.2f}%")
    m3.metric("Parts Placed", f"{placed_count} / {num_items}")
    m4.metric("AI Execution Speed", f"{nest_latency_ms:.2f} ms")

    st.divider()

    # 5. Render High-Resolution Nesting Visualizer Plot
    col_plot, col_table = st.columns([1.3, 1])

    with col_plot:
        st.subheader("🖼️ Nested Sheet Metal Layout")
        fig, ax = plt.subplots(figsize=(8, 8))

        # Sheet background
        ax.add_patch(patches.Rectangle((0, 0), sheet_width, sheet_height, linewidth=2, edgecolor='black', facecolor='#F1F5F9'))

        colors = plt.cm.tab20(np.linspace(0, 1, num_items))

        for idx, manifest_item in enumerate([m for m in placement_manifest if m["Placed"] == "YES"]):
            poly = placed_polygons[idx]
            color = colors[idx % len(colors)]
            is_rot = "90°" in manifest_item["Orientation"]
            edge_c = '#DC2626' if is_rot else '#1E3A8A'

            x_coords, y_coords = poly.exterior.xy
            patch = patches.Polygon(list(zip(x_coords, y_coords)), closed=True, linewidth=1.8, edgecolor=edge_c, facecolor=color, alpha=0.85)
            ax.add_patch(patch)

            cx, cy = poly.centroid.x, poly.centroid.y
            ax.text(cx, cy, f"{manifest_item['Part ID']}\n{manifest_item['Shape']}", color='white', weight='bold', fontsize=8, ha='center', va='center')

        ax.set_xlim(-5, sheet_width + 5)
        ax.set_ylim(-5, sheet_height + 5)
        ax.set_aspect('equal')
        ax.set_title(f"AI Nested Sheet Metal Layout\nUtilization: {utilization_pct:.2f}% | Placed: {placed_count}/{num_items} Parts | Red Edges = 90° Rotated", fontsize=11, fontweight='bold')
        ax.set_xlabel("Width (mm)")
        ax.set_ylabel("Height (mm)")
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        st.pyplot(fig)

    with col_table:
        st.subheader("📋 Part Placement Manifest Table")
        st.dataframe(pd.DataFrame(placement_manifest), use_container_width=True, hide_index=True)
