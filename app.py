"""
Industrial Sheet Metal Nesting AI — Final Production App (`app.py`)

Production-Ready UI supporting:
1. Sheet Stock Dimensions (Width, Height).
2. Custom Drawn Polygon Creator & Part Inventory Table.
3. Preset Puzzles: 5 Unit Squares in 2.8x2.8 Sheet, Factory Order, Custom Drawn Shapes.
4. Multi-Angle AI Nesting Engine (0°, 45°, 90°, 135° Rotations + Shapely 2D Geometry).
5. High-Res Layout Plot, Utilization %, Scrap Ratio, and Placement Manifest.
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
from shapely.affinity import translate, rotate

from extensions.rotation_policy import RotationAttentionPolicy


# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Industrial Metal Nesting AI",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #0F172A; margin-bottom: 0px; }
    .sub-header { font-size: 1.0rem; color: #475569; margin-bottom: 20px; }
    .stButton>button { width: 100%; font-weight: bold; background-color: #0F172A; color: white; border-radius: 8px; height: 48px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🔩 Industrial Sheet Metal Nesting AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Angle Attention Policy + Custom Polygon Creator & Shapely 2D Geometry</div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# Shapely Polygon Shape Library & Parser
# ---------------------------------------------------------
SHAPE_LIBRARY = {
    "Square / Rectangle": lambda w, h: Polygon([(0, 0), (w, 0), (w, h), (0, h)]),
    "L-Shape": lambda w, h: Polygon([(0, 0), (w, 0), (w, h*0.4), (w*0.4, h*0.4), (w*0.4, h), (0, h)]),
    "Triangle": lambda w, h: Polygon([(0, 0), (w, 0), (w*0.5, h)]),
    "T-Shape": lambda w, h: Polygon([(0, h*0.6), (w, h*0.6), (w, h), (w*0.65, h), (w*0.65, 0), (w*0.35, 0), (w*0.35, h), (0, h)]),
    "Trapezoid": lambda w, h: Polygon([(0, 0), (w, 0), (w*0.7, h), (w*0.3, h)])
}

def parse_vertices_string(vert_str: str) -> Polygon:
    """
    Parses a string of (x, y) coordinates into a Shapely Polygon.
    Example: '(0,0), (30,0), (30,15), (0,15)'
    """
    try:
        cleaned = vert_str.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
        nums = [float(val.strip()) for val in cleaned.split(",") if val.strip()]
        coords = [(nums[i], nums[i+1]) for i in range(0, len(nums), 2)]
        if len(coords) >= 3:
            poly = Polygon(coords)
            if poly.is_valid:
                return poly
    except Exception:
        pass
    return Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])


# ---------------------------------------------------------
# Sidebar: Stock Sheet & Rotation Controls
# ---------------------------------------------------------
st.sidebar.header("📐 1. Metal Sheet Stock Size")

if "sheet_w_val" not in st.session_state:
    st.session_state["sheet_w_val"] = 100.0
if "sheet_h_val" not in st.session_state:
    st.session_state["sheet_h_val"] = 100.0

sheet_width = st.sidebar.number_input("Sheet Width (mm)", min_value=1.0, max_value=500.0, value=st.session_state["sheet_w_val"], step=0.1)
sheet_height = st.sidebar.number_input("Sheet Height (mm)", min_value=1.0, max_value=500.0, value=st.session_state["sheet_h_val"], step=0.1)

st.sidebar.divider()
st.sidebar.header("🔄 2. Multi-Angle Rotation Choices")

allow_0 = st.sidebar.checkbox("0° Upright", value=True)
allow_45 = st.sidebar.checkbox("45° Diagonal Tilt (Solves 5-Square Puzzle!)", value=True)
allow_90 = st.sidebar.checkbox("90° Right Angle", value=True)
allow_135 = st.sidebar.checkbox("135° Diagonal", value=True)

candidate_angles = []
if allow_0: candidate_angles.append(0.0)
if allow_45: candidate_angles.append(45.0)
if allow_90: candidate_angles.append(90.0)
if allow_135: candidate_angles.append(135.0)

if len(candidate_angles) == 0:
    candidate_angles = [0.0]


# ---------------------------------------------------------
# Main UI: Parts Inventory & Custom Polygon Drawer
# ---------------------------------------------------------
st.subheader("📦 2. Parts Inventory & Custom Polygon Creator")

tab1, tab2 = st.tabs(["📝 Standard Parts Table & Presets", "✏️ Draw Custom Polygon Vertices"])

with tab1:
    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        if st.button("🧩 Load 5 Unit Squares in 2.8x2.8 Sheet (Friedman Puzzle)"):
            st.session_state["sheet_w_val"] = 2.8
            st.session_state["sheet_h_val"] = 2.8
            st.session_state["parts_df"] = pd.DataFrame([
                {"Part ID": "Square_1", "Shape": "Square / Rectangle", "Width (mm)": 1.0, "Height (mm)": 1.0, "Quantity": 1, "Custom Vertices": ""},
                {"Part ID": "Square_2", "Shape": "Square / Rectangle", "Width (mm)": 1.0, "Height (mm)": 1.0, "Quantity": 1, "Custom Vertices": ""},
                {"Part ID": "Square_3", "Shape": "Square / Rectangle", "Width (mm)": 1.0, "Height (mm)": 1.0, "Quantity": 1, "Custom Vertices": ""},
                {"Part ID": "Square_4", "Shape": "Square / Rectangle", "Width (mm)": 1.0, "Height (mm)": 1.0, "Quantity": 1, "Custom Vertices": ""},
                {"Part ID": "Square_5", "Shape": "Square / Rectangle", "Width (mm)": 1.0, "Height (mm)": 1.0, "Quantity": 1, "Custom Vertices": ""},
            ])
            st.rerun()

    with col_btn2:
        if st.button("🎲 Generate Sample Factory Order"):
            st.session_state["sheet_w_val"] = 100.0
            st.session_state["sheet_h_val"] = 100.0
            st.session_state["parts_df"] = pd.DataFrame([
                {"Part ID": "P1", "Shape": "Square / Rectangle", "Width (mm)": 38.5, "Height (mm)": 39.1, "Quantity": 1, "Custom Vertices": ""},
                {"Part ID": "P2", "Shape": "L-Shape", "Width (mm)": 32.0, "Height (mm)": 35.0, "Quantity": 1, "Custom Vertices": ""},
                {"Part ID": "P3", "Shape": "Triangle", "Width (mm)": 36.0, "Height (mm)": 25.7, "Quantity": 1, "Custom Vertices": ""},
                {"Part ID": "P4", "Shape": "Square / Rectangle", "Width (mm)": 28.0, "Height (mm)": 23.0, "Quantity": 2, "Custom Vertices": ""},
                {"Part ID": "P5", "Shape": "T-Shape", "Width (mm)": 31.2, "Height (mm)": 28.7, "Quantity": 1, "Custom Vertices": ""},
            ])
            st.rerun()

    if "parts_df" not in st.session_state:
        st.session_state["parts_df"] = pd.DataFrame([
            {"Part ID": "Part_1", "Shape": "Square / Rectangle", "Width (mm)": 35.0, "Height (mm)": 25.0, "Quantity": 2, "Custom Vertices": ""},
            {"Part ID": "Part_2", "Shape": "L-Shape", "Width (mm)": 30.0, "Height (mm)": 30.0, "Quantity": 2, "Custom Vertices": ""},
            {"Part ID": "Part_3", "Shape": "Triangle", "Width (mm)": 28.0, "Height (mm)": 20.0, "Quantity": 2, "Custom Vertices": ""},
        ])

    edited_df = st.data_editor(
        st.session_state["parts_df"],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Shape": st.column_config.SelectboxColumn("Shape Type", options=list(SHAPE_LIBRARY.keys()) + ["Custom Drawn Polygon"], required=True),
            "Width (mm)": st.column_config.NumberColumn("Width (mm)", min_value=0.1, max_value=500.0, default=1.0, step=0.1),
            "Height (mm)": st.column_config.NumberColumn("Height (mm)", min_value=0.1, max_value=500.0, default=1.0, step=0.1),
            "Quantity": st.column_config.NumberColumn("Quantity", min_value=1, max_value=10, default=1),
            "Custom Vertices": st.column_config.TextColumn("Custom Vertices (x,y)", help="e.g. (0,0), (40,0), (40,15), (20,15), (20,30), (0,30)")
        }
    )

with tab2:
    st.markdown("### ✏️ Custom Polygon Creator & Live Preview")
    col_draw, col_prev = st.columns([1, 1])

    with col_draw:
        custom_id = st.text_input("Custom Part ID", value="Custom_Shape_1")
        custom_vert_input = st.text_area(
            "Enter Polygon Vertices (X, Y) Coordinates",
            value="(0, 0), (35, 0), (35, 12), (15, 12), (15, 28), (0, 28)",
            help="Enter list of (x, y) coordinate pairs for the polygon vertices."
        )
        custom_qty = st.number_input("Quantity to Add", min_value=1, max_value=10, value=2)

        if st.button("➕ Add Custom Drawn Polygon to Order List"):
            new_row = pd.DataFrame([{
                "Part ID": custom_id,
                "Shape": "Custom Drawn Polygon",
                "Width (mm)": 35.0,
                "Height (mm)": 28.0,
                "Quantity": custom_qty,
                "Custom Vertices": custom_vert_input
            }])
            st.session_state["parts_df"] = pd.concat([edited_df, new_row], ignore_index=True)
            st.success(f"Added '{custom_id}' to the parts inventory!")
            st.rerun()

    with col_prev:
        st.markdown("### 🔍 Shape Geometry Live Preview")
        preview_poly = parse_vertices_string(custom_vert_input)
        fig_prev, ax_prev = plt.subplots(figsize=(4, 4))

        x_coords, y_coords = preview_poly.exterior.xy
        ax_prev.fill(x_coords, y_coords, color='#3B82F6', alpha=0.7, edgecolor='black', linewidth=2)
        ax_prev.scatter(x_coords, y_coords, color='red', zorder=5)

        for i, (x, y) in enumerate(zip(x_coords[:-1], y_coords[:-1])):
            ax_prev.text(x, y, f" P{i+1}\n({x:.0f},{y:.0f})", fontsize=8, fontweight='bold')

        ax_prev.set_aspect('equal')
        ax_prev.set_title(f"Custom Polygon Preview (Area = {preview_poly.area:.1f} mm²)", fontweight='bold', fontsize=10)
        ax_prev.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_prev)

st.divider()

# ---------------------------------------------------------
# Run AI Nesting Engine Button
# ---------------------------------------------------------
if st.button("⚡ EXECUTE MULTI-ANGLE AI NESTING ENGINE"):
    st.markdown("---")

    expanded_items = []
    for idx, row in edited_df.iterrows():
        name = str(row["Part ID"])
        shape_type = str(row["Shape"])
        w = float(row["Width (mm)"])
        h = float(row["Height (mm)"])
        qty = int(row["Quantity"])
        vert_str = str(row.get("Custom Vertices", ""))

        if shape_type == "Custom Drawn Polygon" and len(vert_str) > 5:
            poly = parse_vertices_string(vert_str)
            minx, miny, maxx, maxy = poly.bounds
            w, h = maxx - minx, maxy - miny
        else:
            polygon_fn = SHAPE_LIBRARY.get(shape_type, SHAPE_LIBRARY["Square / Rectangle"])
            poly = polygon_fn(w, h)

        for q in range(qty):
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

    # Pass piece dimensions to Rotation Attention Policy
    piece_features_np = np.array([[item["width"], item["height"]] for item in expanded_items], dtype=np.float32)

    policy = RotationAttentionPolicy(
        input_dim=2,
        d_model=128,
        num_heads=8,
        num_layers=2,
        sheet_width=sheet_width,
        sheet_height=sheet_height
    )
    if os.path.exists("model/scaled_policy.pt"):
        ckpt = torch.load("model/scaled_policy.pt", map_location='cpu')
        policy.load_state_dict(ckpt['model_state_dict'], strict=False)
    policy.eval()

    batch_t = torch.tensor(piece_features_np[np.newaxis, :, :], dtype=torch.float32)
    with torch.no_grad():
        actions_t, _ = policy(batch_t, decode_type="greedy")
        ai_sequence = actions_t[0].cpu().numpy()

    # Geometry Placement Engine
    sheet_poly = box(0, 0, sheet_width, sheet_height)
    sheet_area = sheet_width * sheet_height

    placed_polygons = []
    placement_manifest = []
    used_mask = np.ones(num_items, dtype=bool)

    for action_val in ai_sequence:
        piece_idx = int(action_val % num_items)
        if not used_mask[piece_idx]:
            continue

        item = expanded_items[piece_idx]
        poly = item["polygon"]

        best_placement = None

        for ang in candidate_angles:
            rot_poly = rotate(poly, ang, origin='center') if ang != 0.0 else poly

            minx, miny, maxx, maxy = rot_poly.bounds
            p_w, p_h = maxx - minx, maxy - miny

            candidate_xs = [0.0, sheet_width - p_w, (sheet_width - p_w) / 2.0]
            candidate_ys = [0.0, sheet_height - p_h, (sheet_height - p_h) / 2.0]

            for p in placed_polygons:
                p_minx, p_miny, p_maxx, p_maxy = p.bounds
                candidate_xs.extend([p_maxx, p_minx - p_w, p_maxx - p_w])
                candidate_ys.extend([p_maxy, p_miny - p_h, p_maxy - p_h])

            step_val = 0.05 if sheet_width < 10 else 1.0
            candidate_xs.extend(np.arange(0.0, max(0.0, sheet_width - p_w) + 0.01, step_val))
            candidate_ys.extend(np.arange(0.0, max(0.0, sheet_height - p_h) + 0.01, step_val))

            candidate_xs = sorted(set([x for x in candidate_xs if 0.0 <= x <= sheet_width - p_w + 1e-4]))
            candidate_ys = sorted(set([y for y in candidate_ys if 0.0 <= y <= sheet_height - p_h + 1e-4]))

            grid_candidates = []
            for y in candidate_ys:
                for x in candidate_xs:
                    dist_from_center = abs(x - (sheet_width - p_w)/2.0) + abs(y - (sheet_height - p_h)/2.0)
                    grid_candidates.append((dist_from_center, x, y))

            grid_candidates.sort(key=lambda c_item: -c_item[0])

            for _, x, y in grid_candidates:
                shifted = translate(rot_poly, xoff=x - minx, yoff=y - miny)
                if not sheet_poly.contains(shifted):
                    continue

                overlap = any(shifted.intersects(p) and not shifted.touches(p) for p in placed_polygons)
                if not overlap:
                    best_placement = (x, y, ang, shifted)
                    break

            if best_placement is not None:
                break

        used_mask[piece_idx] = False

        if best_placement is not None:
            px, py, ang_used, final_poly = best_placement
            placed_polygons.append(final_poly)
            placement_manifest.append({
                "Part ID": item["id"],
                "Shape": item["shape"],
                "Placed": "YES",
                "Position (X, Y)": f"({px:.2f}, {py:.2f})",
                "Rotation Angle": f"{ang_used:.0f}°",
                "Area (mm²)": f"{item['area']:.2f}"
            })
        else:
            placement_manifest.append({
                "Part ID": item["id"],
                "Shape": item["shape"],
                "Placed": "NO (Sheet Full)",
                "Position (X, Y)": "-",
                "Rotation Angle": "-",
                "Area (mm²)": f"{item['area']:.2f}"
            })

    nest_latency_ms = (time.time() - t_start) * 1000.0

    total_placed_area = sum(p.area for p in placed_polygons)
    utilization_pct = (total_placed_area / sheet_area) * 100.0
    scrap_pct = 100.0 - utilization_pct
    placed_count = len(placed_polygons)

    # Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sheet Utilization", f"{utilization_pct:.2f}%")
    m2.metric("Scrap Material Ratio", f"{scrap_pct:.2f}%")
    m3.metric("Parts Placed", f"{placed_count} / {num_items}")
    m4.metric("AI Execution Speed", f"{nest_latency_ms:.2f} ms")

    st.divider()

    # Visualizer & Manifest Table
    col_plot, col_table = st.columns([1.3, 1])

    with col_plot:
        st.subheader("🖼️ Nested Sheet Metal Layout")
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.add_patch(patches.Rectangle((0, 0), sheet_width, sheet_height, linewidth=2, edgecolor='black', facecolor='#F1F5F9'))

        colors = plt.cm.tab20(np.linspace(0, 1, num_items))

        for idx, manifest_item in enumerate([m for m in placement_manifest if m["Placed"] == "YES"]):
            poly = placed_polygons[idx]
            color = colors[idx % len(colors)]
            ang_str = manifest_item["Rotation Angle"]
            edge_c = '#DC2626' if ang_str != "0°" else '#1E3A8A'

            x_coords, y_coords = poly.exterior.xy
            patch = patches.Polygon(list(zip(x_coords, y_coords)), closed=True, linewidth=1.8, edgecolor=edge_c, facecolor=color, alpha=0.85)
            ax.add_patch(patch)

            cx, cy = poly.centroid.x, poly.centroid.y
            ax.text(cx, cy, f"{manifest_item['Part ID']}\n({ang_str})", color='white', weight='bold', fontsize=8, ha='center', va='center')

        ax.set_xlim(-0.1 * sheet_width, 1.1 * sheet_width)
        ax.set_ylim(-0.1 * sheet_height, 1.1 * sheet_height)
        ax.set_aspect('equal')
        ax.set_title(f"Multi-Angle AI Nested Sheet Metal Layout\nUtilization: {utilization_pct:.2f}% | Placed: {placed_count}/{num_items} Parts | Red Edges = Rotated Parts", fontsize=11, fontweight='bold')
        ax.set_xlabel("Width (mm)")
        ax.set_ylabel("Height (mm)")
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        st.pyplot(fig)

    with col_table:
        st.subheader("📋 Part Placement Manifest Table")
        st.dataframe(pd.DataFrame(placement_manifest), use_container_width=True, hide_index=True)
