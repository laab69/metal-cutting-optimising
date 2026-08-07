"""
Industrial Sheet Metal Nesting AI — Final Production App (`app.py`)

Production-Ready UI supporting:
1. Sheet Stock Dimensions (Width, Height).
2. Interactive Mouse Canvas Polygon Drawer (`streamlit-drawable-canvas`).
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
from streamlit_drawable_canvas import st_canvas

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
st.markdown('<div class="sub-header">Multi-Angle Attention Policy + Interactive Mouse Polygon Drawer & Shapely Geometry</div>', unsafe_allow_html=True)


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
# Main UI: Parts Inventory & Mouse Polygon Drawer
# ---------------------------------------------------------
st.subheader("📦 2. Parts Inventory & Mouse Polygon Canvas Drawer")

tab1, tab2, tab3 = st.tabs(["📝 Standard Parts Table & Presets", "🖱️ Draw Custom Polygon with Mouse", "✏️ Coordinate & Preset Designer"])

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
    st.markdown("### 🖱️ Mouse Canvas Polygon Drawer")
    st.caption("Click and drag your mouse directly on the white canvas below to draw your custom shape!")

    col_canvas, col_mouse_info = st.columns([1.3, 1])

    with col_canvas:
        drawing_mode = st.radio("Mouse Tool Mode:", ("freedraw", "polygon", "rect"), horizontal=True)

        canvas_result = st_canvas(
            fill_color="rgba(59, 130, 246, 0.4)",
            stroke_width=3,
            stroke_color="#0F172A",
            background_color="#FFFFFF",
            height=350,
            width=450,
            drawing_mode=drawing_mode,
            update_streamlit=True,
            key="mouse_canvas_drawing",
        )

    with col_mouse_info:
        st.markdown("### 📋 Mouse Shape Status")
        mouse_part_id = st.text_input("Custom Mouse Part ID", value="Mouse_Drawn_Part_1")
        mouse_qty = st.number_input("Quantity to Add to Order", min_value=1, max_value=10, value=2)

        coords_str = ""
        extracted_coords = []

        if canvas_result.json_data is not None:
            objects = canvas_result.json_data.get("objects", [])
            if len(objects) > 0:
                last_obj = objects[-1]
                obj_type = last_obj.get("type")

                if obj_type == "path":
                    path_data = last_obj.get("path", [])
                    for p in path_data:
                        if len(p) >= 3 and p[0] in ["M", "L", "Q", "C"]:
                            extracted_coords.append((round(float(p[1]) / 10.0, 1), round((350.0 - float(p[2])) / 10.0, 1)))

                elif obj_type == "polygon":
                    points = last_obj.get("points", [])
                    for pt in points:
                        extracted_coords.append((round(float(pt["x"]) / 10.0, 1), round((350.0 - float(pt["y"])) / 10.0, 1)))

                elif obj_type == "rect":
                    left = round(float(last_obj.get("left", 0)) / 10.0, 1)
                    top = round((350.0 - float(last_obj.get("top", 0))) / 10.0, 1)
                    rw = round(float(last_obj.get("width", 20)) / 10.0, 1)
                    rh = round(float(last_obj.get("height", 20)) / 10.0, 1)
                    extracted_coords = [
                        (left, top - rh),
                        (left + rw, top - rh),
                        (left + rw, top),
                        (left, top)
                    ]

        if len(extracted_coords) >= 3:
            coords_str = ", ".join([f"({x},{y})" for x, y in extracted_coords])
            st.success(f"Detected {len(extracted_coords)} Vertices from Mouse Drawing!")
            st.code(coords_str, language="text")

            if st.button("➕ Add Mouse Drawn Polygon to Order List"):
                new_row = pd.DataFrame([{
                    "Part ID": mouse_part_id,
                    "Shape": "Custom Drawn Polygon",
                    "Width (mm)": 30.0,
                    "Height (mm)": 30.0,
                    "Quantity": mouse_qty,
                    "Custom Vertices": coords_str
                }])
                st.session_state["parts_df"] = pd.concat([edited_df, new_row], ignore_index=True)
                st.success(f"Added Mouse Drawn Polygon '{mouse_part_id}' to order inventory!")
                st.rerun()
        else:
            st.info("Draw a shape on the left white canvas box with your mouse!")

with tab3:
    st.markdown("### ✏️ Coordinate & Preset Designer")
    col_designer, col_live_prev = st.columns([1.1, 1])

    with col_designer:
        preset_choice = st.selectbox(
            "Quick Load Custom Polygon Preset:",
            ["Custom (Manual Points)", "L-Bracket Plate", "Notched Rectangle", "Triangle Gusset", "Regular Hexagon"]
        )

        default_points = "(0,0), (35,0), (35,12), (15,12), (15,28), (0,28)"
        if preset_choice == "L-Bracket Plate":
            default_points = "(0,0), (40,0), (40,15), (15,15), (15,40), (0,40)"
        elif preset_choice == "Notched Rectangle":
            default_points = "(0,0), (50,0), (50,30), (35,30), (35,20), (15,20), (15,30), (0,30)"
        elif preset_choice == "Triangle Gusset":
            default_points = "(0,0), (45,0), (22.5,35)"
        elif preset_choice == "Regular Hexagon":
            default_points = "(10,0), (30,0), (40,17.3), (30,34.6), (10,34.6), (0,17.3)"

        custom_vert_text = st.text_area("Vertex Coordinates (X, Y):", value=default_points, height=100)
        custom_shape_name = st.text_input("Part Name:", value="Custom_Plate_1", key="t3_name")
        custom_shape_qty = st.number_input("Quantity:", min_value=1, max_value=10, value=2, key="t3_qty")

    with col_live_prev:
        st.markdown("#### Live Geometry Preview")
        preview_polygon = parse_vertices_string(custom_vert_text)

        fig_prev, ax_prev = plt.subplots(figsize=(4.5, 4))
        if preview_polygon is not None and preview_polygon.is_valid:
            x_c, y_c = preview_polygon.exterior.xy
            ax_prev.fill(x_c, y_c, color='#3B82F6', alpha=0.7, edgecolor='#1E3A8A', linewidth=2)
            ax_prev.scatter(x_c, y_c, color='#DC2626', zorder=5, s=40)
            ax_prev.set_title(f"Area: {preview_polygon.area:.1f} mm²", fontsize=10, fontweight='bold')
        ax_prev.set_aspect('equal')
        ax_prev.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_prev)

    if st.button("➕ Add Preset Polygon to Order List"):
        if preview_polygon is not None and preview_polygon.is_valid:
            minx, miny, maxx, maxy = preview_polygon.bounds
            new_row = pd.DataFrame([{
                "Part ID": custom_shape_name,
                "Shape": "Custom Drawn Polygon",
                "Width (mm)": round(maxx - minx, 1),
                "Height (mm)": round(maxy - miny, 1),
                "Quantity": custom_shape_qty,
                "Custom Vertices": custom_vert_text
            }])
            st.session_state["parts_df"] = pd.concat([edited_df, new_row], ignore_index=True)
            st.success(f"Added '{custom_shape_name}' ({custom_shape_qty}x) to order inventory!")
            st.rerun()

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
