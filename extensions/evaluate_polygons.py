"""
Shapely Irregular Polygon Nesting Evaluator (`evaluate_polygons.py`)

WHY THIS SCRIPT EXISTS:
Runs and visualizes irregular polygon nesting (L-shapes, Triangles, T-shapes, Trapezoids) 
using Shapely exact 2D geometric intersection calculations.

Outputs:
1. Utilization metrics comparing Random Selection vs. Largest-Area-First for irregular shapes.
2. Visual plot 'polygon_nesting_layout.png' displaying exact Shapely polygon layouts.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon
from extensions.polygon_env import PolygonNestingEnv, generate_polygon_instance


def run_polygon_extension(
    num_pieces: int = 10,
    sheet_width: float = 100.0,
    sheet_height: float = 100.0,
    seed: int = 42,
    save_plot_path: str = "polygon_nesting_layout.png"
):
    print("=" * 70)
    print("  PHASE F (EXTENSION 2): IRREGULAR POLYGON NESTING (SHAPELY)")
    print("=" * 70)

    env = PolygonNestingEnv(sheet_width=sheet_width, sheet_height=sheet_height, num_pieces=num_pieces)
    polys = generate_polygon_instance(num_pieces=num_pieces, seed=seed)

    print(f"Generated instance with {num_pieces} irregular Shapely polygons.")
    print("Polygon types: L-Shapes, Triangles, T-Shapes, Trapezoids, Rectangles.")
    print("-" * 70)

    # 1. Evaluate Largest-Area-First Heuristic for Irregular Polygons
    t0 = time.time()
    state = env.reset(polygons=polys)

    # Sort polygons by Shapely area in descending order
    areas = [p.area for p in polys]
    sorted_indices = np.argsort(-np.array(areas))

    for act in sorted_indices:
        state, reward, done, info = env.step(act)

    poly_utilization = env.score()
    elapsed_t = time.time() - t0
    placed_count = len(env.placed_polygons)

    print(f"RESULTS (Largest-Area-First Polygon Nesting):")
    print(f"  Total Polygons Placed : {placed_count} / {num_pieces}")
    print(f"  Final Utilization     : {poly_utilization:.2f}%")
    print(f"  Execution Wall-Clock  : {elapsed_t:.3f} seconds")
    print("=" * 70)

    # 2. Visualize Irregular Shapely Polygon Layout
    fig, ax = plt.subplots(figsize=(8, 8))

    # Sheet background
    sheet_rect = patches.Rectangle((0, 0), sheet_width, sheet_height, linewidth=2, edgecolor='black', facecolor='#f8f9fa')
    ax.add_patch(sheet_rect)

    colors = plt.cm.Set3(np.linspace(0, 1, num_pieces))

    for idx, poly in enumerate(env.placed_polygons):
        orig_idx = env.placed_indices[idx]
        color = colors[idx % len(colors)]

        # Extract polygon outer boundary coordinates
        x_coords, y_coords = poly.exterior.xy
        vertices = list(zip(x_coords, y_coords))

        polygon_patch = patches.Polygon(vertices, closed=True, linewidth=1.5, edgecolor='darkgreen', facecolor=color, alpha=0.85)
        ax.add_patch(polygon_patch)

        # Label centroid
        cx, cy = poly.centroid.x, poly.centroid.y
        ax.text(cx, cy, f"P{orig_idx}\nArea={poly.area:.0f}", color='black', weight='bold', fontsize=8, ha='center', va='center')

    ax.set_xlim(-5, sheet_width + 5)
    ax.set_ylim(-5, sheet_height + 5)
    ax.set_aspect('equal')
    ax.set_title(f"Phase F: Irregular Polygon Nesting (Shapely Geometry)\nPlaced: {placed_count}/{num_pieces} | Utilization: {poly_utilization:.2f}%", fontsize=11, fontweight='bold')
    ax.set_xlabel("X (Sheet Width)")
    ax.set_ylabel("Y (Sheet Height)")
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_plot_path, dpi=150)
    print(f"Irregular polygon layout plot saved to '{os.path.abspath(save_plot_path)}'\n")
    plt.close()

    return poly_utilization

if __name__ == "__main__":
    run_polygon_extension()
