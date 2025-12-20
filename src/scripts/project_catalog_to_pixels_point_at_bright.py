# src/scripts/project_catalog_to_pixels_point_at_bright.py
import csv, math, numpy as np, argparse
from pathlib import Path

# ---------------- CLI ----------------
ap = argparse.ArgumentParser()
ap.add_argument("--out_dir", required=True)
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--height", type=int, default=480)
ap.add_argument("--fov_x_deg", type=float, default=30.0)
args = ap.parse_args()

OUT_DIR = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

WIDTH  = args.width
HEIGHT = args.height
FOV_X_DEG = args.fov_x_deg

# ---------------- TRUE ATTITUDE ----------------
R_TRUE = np.eye(3)
np.savetxt(OUT_DIR / "R_true.txt", R_TRUE)

# ---------------- INTRINSICS ----------------
fx = (WIDTH / 2) / math.tan(math.radians(FOV_X_DEG / 2))
fy = fx
cx = WIDTH / 2
cy = HEIGHT / 2

# ---------------- INPUT CATALOG ----------------
ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "data/catalog_unit_vectors.csv"

rows = []

with open(CATALOG, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        rows.append([
            int(row["id"]),                     # ★ CRITICAL
            float(row["x"]),
            float(row["y"]),
            float(row["z"]),
            float(row["mag"])
        ])

# ---------------- PROJECT ----------------
out_rows = []

for star_id, x, y, z, mag in rows:
    v_inertial = np.array([x, y, z])
    v_cam = R_TRUE @ v_inertial

    if v_cam[2] <= 0:
        continue

    u = fx * (v_cam[0] / v_cam[2]) + cx
    v = fy * (v_cam[1] / v_cam[2]) + cy

    if 0 <= u < WIDTH and 0 <= v < HEIGHT:
        out_rows.append([star_id, u, v, mag, x, y, z])

# ---------------- SAVE ----------------
with open(OUT_DIR / "catalog_projected.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["star_id", "u", "v", "mag", "x", "y", "z"])
    w.writerows(out_rows)

print("✓ Projection complete")
print("✓ R_true saved")
print(f"✓ Stars in FOV: {len(out_rows)}")
