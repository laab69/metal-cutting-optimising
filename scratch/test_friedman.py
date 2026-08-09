import shapely
from shapely.geometry import Polygon, box
from shapely.affinity import translate, rotate

sheet_w, sheet_h = 2.8, 2.8
sheet_poly = box(0, 0, sheet_w, sheet_h)

# 5 unit squares
unit_sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

# Placement logic testing centroid rotation & exact corner alignment
placed_polys = []

# 4 Corner positions
corners = [
    (0.0, 0.0),                  # Bottom-Left corner
    (sheet_w - 1.0, 0.0),         # Bottom-Right corner (x=1.8)
    (0.0, sheet_h - 1.0),         # Top-Left corner (y=1.8)
    (sheet_w - 1.0, sheet_h - 1.0) # Top-Right corner (1.8, 1.8)
]

for cx, cy in corners:
    placed = translate(unit_sq, xoff=cx, yoff=cy)
    placed_polys.append(placed)

# 5th square rotated 45 degrees around center!
sq5_rotated = rotate(unit_sq, 45, origin='center')
minx, miny, maxx, maxy = sq5_rotated.bounds
# Shift centroid of 5th square to center of sheet (1.4, 1.4)
sq5_center_x, sq5_center_y = 1.4, 1.4
sq5_placed = translate(sq5_rotated, xoff=sq5_center_x - (minx + maxx)/2.0, yoff=sq5_center_y - (miny + maxy)/2.0)

# Check overlap
overlaps = any(sq5_placed.intersects(p) and not sq5_placed.touches(p) for p in placed_polys)
contains = sheet_poly.contains(sq5_placed)

print(f"5th Square Inside Sheet (2.8x2.8): {contains}")
print(f"5th Square Overlaps Corner Squares: {overlaps}")
if contains and not overlaps:
    placed_polys.append(sq5_placed)
    print(f"SUCCESS! All {len(placed_polys)} squares placed cleanly!")
