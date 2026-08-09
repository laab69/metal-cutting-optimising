import shapely
from shapely.geometry import Polygon, box
from shapely.affinity import translate, rotate
import numpy as np

sheet_w, sheet_h = 2.8, 2.8
sheet_poly = box(0, 0, sheet_w, sheet_h)

unit_sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
candidate_angles = [0.0, 45.0, 90.0, 135.0]

placed_polygons = []

# 5 unit squares
expanded_items = [{"id": f"Square_{i+1}", "polygon": unit_sq} for i in range(5)]

for item in expanded_items:
    poly = item["polygon"]
    best_placement = None

    for ang in candidate_angles:
        rot_poly = rotate(poly, ang, origin='center') if ang != 0.0 else poly

        minx, miny, maxx, maxy = rot_poly.bounds
        p_w, p_h = maxx - minx, maxy - miny

        candidate_xs = [0.0, sheet_w - p_w, (sheet_w - p_w) / 2.0]
        candidate_ys = [0.0, sheet_h - p_h, (sheet_h - p_h) / 2.0]

        for p in placed_polygons:
            p_minx, p_miny, p_maxx, p_maxy = p.bounds
            candidate_xs.extend([p_maxx, p_minx - p_w, p_maxx - p_w])
            candidate_ys.extend([p_maxy, p_miny - p_h, p_maxy - p_h])

        step_val = 0.05
        candidate_xs.extend(np.arange(0.0, max(0.0, sheet_w - p_w) + 0.01, step_val))
        candidate_ys.extend(np.arange(0.0, max(0.0, sheet_h - p_h) + 0.01, step_val))

        candidate_xs = sorted(set([x for x in candidate_xs if 0.0 <= x <= sheet_w - p_w + 1e-4]))
        candidate_ys = sorted(set([y for y in candidate_ys if 0.0 <= y <= sheet_h - p_h + 1e-4]))

        for y in candidate_ys:
            for x in candidate_xs:
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
            break

    if best_placement is not None:
        px, py, ang_used, final_poly = best_placement
        placed_polygons.append(final_poly)
        print(f"Placed {item['id']} at ({px:.2f}, {py:.2f}) with angle {ang_used}°")
    else:
        print(f"FAILED to place {item['id']}")

print(f"\nTotal Placed: {len(placed_polygons)} / 5")
