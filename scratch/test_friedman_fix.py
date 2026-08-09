import shapely
from shapely.geometry import Polygon, box
from shapely.affinity import translate, rotate
import numpy as np

sheet_w, sheet_h = 2.8, 2.8
sheet_poly = box(0, 0, sheet_w, sheet_h)
unit_sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

# Placement Engine with Friedman Corner Strategy
# For square containers when packing multiple identical unit squares:
# Candidate positions should prioritize corners of the container!

def try_friedman_placement():
    placed_polygons = []
    expanded_items = [{"id": f"Square_{i+1}", "polygon": unit_sq} for i in range(5)]

    # Candidate origins include container corners!
    corner_origins = [
        (0.0, 0.0),
        (sheet_w - 1.0, 0.0),
        (0.0, sheet_h - 1.0),
        (sheet_w - 1.0, sheet_h - 1.0),
        ((sheet_w - 1.0)/2.0, (sheet_h - 1.0)/2.0)
    ]

    for item in expanded_items:
        poly = item["polygon"]
        best_placement = None

        for ang in [0.0, 45.0, 90.0, 135.0]:
            rot_poly = rotate(poly, ang, origin='center') if ang != 0.0 else poly
            minx, miny, maxx, maxy = rot_poly.bounds
            p_w, p_h = maxx - minx, maxy - miny

            # Candidate search locations: corner origins first!
            candidate_xs = [0.0, sheet_w - p_w, (sheet_w - p_w) / 2.0]
            candidate_ys = [0.0, sheet_h - p_h, (sheet_h - p_h) / 2.0]

            for p in placed_polygons:
                p_minx, p_miny, p_maxx, p_maxy = p.bounds
                candidate_xs.extend([p_maxx, p_minx - p_w, p_maxx - p_w])
                candidate_ys.extend([p_maxy, p_miny - p_h, p_maxy - p_h])

            # Filter valid coordinates
            candidate_xs = sorted(set([x for x in candidate_xs if 0.0 <= x <= sheet_w - p_w + 1e-4]))
            candidate_ys = sorted(set([y for y in candidate_ys if 0.0 <= y <= sheet_h - p_h + 1e-4]))

            # Sort candidate (y, x) pairs: for 0° squares, prioritize four corners (0,0), (1.8,0), (0,1.8), (1.8,1.8)!
            # Sort order: max distance from sheet center!
            grid_candidates = []
            for y in candidate_ys:
                for x in candidate_xs:
                    # Distance from center of search area
                    dist_from_center = abs(x - (sheet_w - p_w)/2.0) + abs(y - (sheet_h - p_h)/2.0)
                    grid_candidates.append((dist_from_center, x, y))

            # Sort candidate positions by HIGHEST distance from center first (outer corners first!)
            grid_candidates.sort(key=lambda item: -item[0])

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

        if best_placement is not None:
            px, py, ang_used, final_poly = best_placement
            placed_polygons.append(final_poly)
            print(f"Placed {item['id']} at ({px:.2f}, {py:.2f}) with angle {ang_used}°")

    print(f"\nTotal Placed: {len(placed_polygons)} / 5")

try_friedman_placement()
