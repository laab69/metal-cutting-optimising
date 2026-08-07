# Phase F Extensions: Rotation & Irregular Polygons (`extensions/`)

Welcome to **Phase F**! In this extension phase, we expand the problem formulation beyond fixed-orientation rectangles into real-world manufacturing constraints:

1. **Rotation Action Space ($2N$ Actions)**: Allowing pieces to be rotated by 90°.
2. **Irregular Polygon Shapes (`Shapely`)**: Handling non-rectangular sheet metal geometries.

---

## 🔄 1. Rotation Action Space ($2N$ Candidate Actions)

### Problem Formulation
In real sheet metal cutting, rectangular parts can often be rotated by $90^\circ$ to fit into narrow gaps on the sheet.

* **Action Space**: For $N$ remaining pieces, the action space expands from $N$ to $2N$ choices:
  * Actions $0 \dots N-1$: Place piece $i$ in **Original Orientation ($0^\circ$)** with dimensions $(w_i, h_i)$.
  * Actions $N \dots 2N-1$: Place piece $i-N$ in **Rotated Orientation ($90^\circ$)** with dimensions $(h_{i-N}, w_{i-N})$.

### Dual Action Masking Rule
When piece $k$ is selected in *either* orientation:
* We set `mask[k] = False` ($0^\circ$ candidate disabled).
* We set `mask[k + N] = False` ($90^\circ$ candidate disabled).
* **Why**: A physical sheet metal part can only be cut once. Masking both orientations prevents double-placement.

---

## 🔷 2. Irregular Polygon Nesting (`Shapely`)

For non-rectangular shapes (L-shapes, triangles, concave parts):
* Shapes are represented as 2D `Polygon` objects using the `Shapely` computational geometry library.
* Placement utilizes minimum bounding boxes, centroid offsets, and exact polygon overlap checking (`poly1.intersects(poly2)`).

---

## 📁 File Structure

* [`rotation_env.py`](file:///c:/Users/C12H28O9/OneDrive/Desktop/metal%20cutting%20optmising/extensions/rotation_env.py): Environment supporting $2N$ rotation action space and dual masking.
* [`rotation_policy.py`](file:///c:/Users/C12H28O9/OneDrive/Desktop/metal%20cutting%20optmising/extensions/rotation_policy.py): Rotation-aware Attention Policy Network outputting $2N$ action logits.
* [`evaluate_rotation.py`](file:///c:/Users/C12H28O9/OneDrive/Desktop/metal%20cutting%20optmising/extensions/evaluate_rotation.py): Benchmarks training progress and exports visual layout plots comparing fixed vs. 90°-rotated packing.
* [`polygon_env.py`](file:///c:/Users/C12H28O9/OneDrive/Desktop/metal%20cutting%20optmising/extensions/polygon_env.py): Shapely-based irregular polygon environment and visualizer.

---

## 🚀 How to Run the Rotation Extension

```bash
python -m extensions.evaluate_rotation
```
This script trains and evaluates the rotation-aware policy, exporting layout visualization to `rotation_nesting_comparison.png`.
