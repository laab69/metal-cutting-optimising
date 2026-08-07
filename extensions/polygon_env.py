"""
Shapely Irregular Polygon Nesting Environment (`polygon_env.py`)

WHY THIS MODULE EXISTS:
Real-world sheet metal fabrication requires cutting irregular 2D polygons (L-shapes, 
triangles, T-bars, trapezoids) rather than simple axis-aligned rectangles.

This environment uses Shapely to:
1. Represent arbitrary 2D polygonal geometry via shapely.geometry.Polygon.
2. Enforce exact spatial containment within sheet boundaries (sheet.contains(poly)).
3. Perform exact polygon-polygon intersection checks (poly1.intersects(poly2)).
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import shapely
from shapely.geometry import Polygon, box
from shapely.affinity import translate


# Pre-defined templates for irregular 2D polygon shapes (L-shape, Triangle, T-shape, Trapezoid)
POLYGON_TEMPLATES = [
    # L-Shape
    Polygon([(0, 0), (25, 0), (25, 10), (10, 10), (10, 25), (0, 25)]),
    # Triangle
    Polygon([(0, 0), (30, 0), (15, 25)]),
    # T-Shape
    Polygon([(0, 15), (30, 15), (30, 25), (20, 25), (20, 0), (10, 0), (10, 25), (0, 25)]),
    # Trapezoid
    Polygon([(0, 0), (30, 0), (20, 20), (5, 20)]),
    # Rectangle
    Polygon([(0, 0), (25, 0), (25, 15), (0, 15)])
]


def generate_polygon_instance(num_pieces: int = 10, seed: Optional[int] = None) -> List[Polygon]:
    """
    Generates an instance of random irregular polygons selected from templates.
    """
    if seed is not None:
        np.random.seed(seed)

    selected_polys = []
    for _ in range(num_pieces):
        template_idx = np.random.choice(len(POLYGON_TEMPLATES))
        poly = POLYGON_TEMPLATES[template_idx]
        selected_polys.append(poly)

    return selected_polys


class PolygonNestingEnv:
    def __init__(self, sheet_width: float = 100.0, sheet_height: float = 100.0, num_pieces: int = 10):
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height
        self.sheet_poly = box(0, 0, sheet_width, sheet_height)
        self.sheet_area = sheet_width * sheet_height
        self.num_pieces = num_pieces

        self.polygons: List[Polygon] = []
        self.mask: Optional[np.ndarray] = None
        self.placed_polygons: List[Polygon] = []
        self.placed_indices: List[int] = []
        self.step_count: int = 0

    def reset(self, polygons: Optional[List[Polygon]] = None, seed: Optional[int] = None) -> Dict[str, Any]:
        if polygons is not None:
            self.polygons = list(polygons)
            self.num_pieces = len(polygons)
        else:
            self.polygons = generate_polygon_instance(num_pieces=self.num_pieces, seed=seed)

        self.mask = np.ones(self.num_pieces, dtype=bool)
        self.placed_polygons = []
        self.placed_indices = []
        self.step_count = 0

        return self._get_state()

    def _get_state(self) -> Dict[str, Any]:
        return {
            "mask": self.mask.copy(),
            "placed_polygons": list(self.placed_polygons),
            "step_count": self.step_count,
            "utilization": self.compute_utilization()
        }

    def place_bottom_left_polygon(self, poly: Polygon) -> Optional[Polygon]:
        """
        Searches candidate (x, y) coordinates to place an irregular Shapely polygon on the sheet.
        """
        minx, miny, maxx, maxy = poly.bounds
        poly_w = maxx - minx
        poly_h = maxy - miny

        # Candidate grid search steps
        grid_step = 2.0
        candidate_xs = np.arange(0.0, self.sheet_width - poly_w + 1.0, grid_step)
        candidate_ys = np.arange(0.0, self.sheet_height - poly_h + 1.0, grid_step)

        for y in candidate_ys:
            for x in candidate_xs:
                # Shift polygon to candidate (x, y) position
                shifted_poly = translate(poly, xoff=x - minx, yoff=y - miny)

                # Check 1: Entire polygon inside sheet
                if not self.sheet_poly.contains(shifted_poly):
                    continue

                # Check 2: No overlap with any previously placed polygon
                overlap = False
                for placed_p in self.placed_polygons:
                    if shifted_poly.intersects(placed_p) and not shifted_poly.touches(placed_p):
                        overlap = True
                        break

                if not overlap:
                    return shifted_poly

        return None

    def step(self, action_idx: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if not self.mask[action_idx]:
            raise ValueError(f"Action Error: Polygon {action_idx} is already placed!")

        poly = self.polygons[action_idx]
        placed_poly = self.place_bottom_left_polygon(poly)

        self.mask[action_idx] = False
        self.step_count += 1

        placed_successfully = False
        if placed_poly is not None:
            self.placed_polygons.append(placed_poly)
            self.placed_indices.append(action_idx)
            placed_successfully = True

        done = (self.step_count == self.num_pieces)
        reward = self.score() if done else 0.0

        info = {
            "placed_successfully": placed_successfully,
            "action_idx": action_idx
        }

        return self._get_state(), reward, done, info

    def compute_utilization(self) -> float:
        total_placed_area = sum(p.area for p in self.placed_polygons)
        return float(total_placed_area / self.sheet_area)

    def score(self) -> float:
        return self.compute_utilization() * 100.0
