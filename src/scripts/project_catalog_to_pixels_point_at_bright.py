# src/scripts/project_catalog_to_pixels_point_at_bright.py
"""
Project catalog unit vectors into pixel coords.
This script automatically chooses a target direction by averaging the brightest K stars
and rotates the catalog so the camera looks at that patch. This avoids the 'zero visible'
problem when the camera initially faces an empty direction.
"""
import sys, pathlib, os, csv, math
project_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import numpy as np
from src.utils.camera_model import make_cam_params, direction_to_pixel

# ---------- user-tweakable parameters ----------
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FOV_X = 30.0        # horizontal FOV in degrees (can increase if you want)
BRIGHT_K = 200          # number of brightest stars to average for target direction
INFILE = "data/catalog_unit_vectors.csv"
OUTFILE = "results/catalog_projected.csv"
# ------------------------------------------------

os.makedirs("results", exist_ok=True)
cam = make_cam_params(width=CAM_WIDTH, height=CAM_HEIGHT, fov_x=CAM_FOV_X)

# read catalog into memory (we need mag + x,y,z)
rows = []
with open(INFILE, newline='', encoding='utf-8') as f:
    r = csv.DictReader(f)
    for row in r:
        try:
            x = float(row['x']); y = float(row['y']); z = float(row['z'])
        except Exception:
            continue
        mag = None
        if row.get('mag') not in (None, ''):
            try:
                mag = float(row['mag'])
            except:
                mag = None
        rows.append({'row': row, 'x': x, 'y': y, 'z': z, 'mag': mag})

if len(rows) == 0:
    print("No catalog rows found in", INFILE)
    sys.exit(1)

# pick brightest K (smallest mag). If many mag are missing, fallback to first N vectors.
with_mag = [r for r in rows if r['mag'] is not None]
if len(with_mag) >= 10:
    with_mag.sort(key=lambda r: r['mag'])
    pick = with_mag[:min(BRIGHT_K, len(with_mag))]
else:
    # fallback: take the first BRIGHT_K rows
    pick = rows[:min(BRIGHT_K, len(rows))]

# compute average vector direction (weighted equally)
vecs = np.array([[p['x'], p['y'], p['z']] for p in pick], dtype=float)
mean_vec = vecs.mean(axis=0)
norm = np.linalg.norm(mean_vec)
if norm < 1e-12:
    print("Couldn't compute a stable target direction; aborting.")
    sys.exit(1)
target = mean_vec / norm
print("Auto target direction (unit):", target.tolist())

# rotation: map 'target' -> +Z (0,0,1)
def rotation_matrix_from_vectors(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s < 1e-12:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    R = np.eye(3) + K + K @ K * ((1 - c) / (s**2))
    return R

R = rotation_matrix_from_vectors(target, np.array([0.0, 0.0, 1.0]))

# project all rows (rotate first)
total = 0
visible = 0
sample = []
with open(OUTFILE, "w", newline='', encoding='utf-8') as fout:
    w = csv.writer(fout)
    w.writerow(["id","ra_deg","dec_deg","mag","x","y","z","u","v"])
    for entry in rows:
        total += 1
        vin = np.array([entry['x'], entry['y'], entry['z']], dtype=float)
        vcam = R.dot(vin)   # inertial -> camera frame
        vec = (float(vcam[0]), float(vcam[1]), float(vcam[2]))
        pix = direction_to_pixel(vec, cam)
        if pix is not None:
            visible += 1
            u, v = pix
            r = entry['row']
            w.writerow([r.get('id',''), r.get('ra_deg',''), r.get('dec_deg',''), r.get('mag',''),
                        f"{entry['x']:.12g}", f"{entry['y']:.12g}", f"{entry['z']:.12g}", f"{u:.3f}", f"{v:.3f}"])
            if len(sample) < 8:
                sample.append((r.get('id',''), r.get('ra_deg',''), r.get('dec_deg',''), f"{u:.3f}", f"{v:.3f}"))

print(f"Total catalog rows: {total}")
print(f"Visible (in cam FOV): {visible}")
print("Wrote:", OUTFILE)
print("Camera params:", cam)
if sample:
    print("Sample projected rows (id, ra_deg, dec_deg, u, v):")
    for s in sample:
        print(" ", s)
