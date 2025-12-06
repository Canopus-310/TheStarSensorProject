# scripts/project_catalog_to_pixels.py

# --- begin: make project root importable ---
import sys, pathlib
project_root = pathlib.Path(__file__).resolve().parents[2]  # two levels up from src/scripts -> project root
sys.path.insert(0, str(project_root))
# --- end ---
import csv
from src.utils.camera_model import make_cam_params, direction_to_pixel
import os

os.makedirs("results", exist_ok=True)

# camera params (change as you like)
cam = make_cam_params(width=640, height=480, fov_x=30.0)

inf = "C:/Users/aryan/PycharmProjects/TheStarSensorProject/data/catalog_unit_vectors.csv"
out = "C:/Users/aryan/PycharmProjects/TheStarSensorProject/results/catalog_projected.csv"

with open(inf, newline='', encoding='utf-8') as f_in, open(out, "w", newline='', encoding='utf-8') as f_out:
    r = csv.DictReader(f_in)
    w = csv.writer(f_out)
    w.writerow(["id", "ra_deg", "dec_deg", "mag", "x", "y", "z", "u", "v"])
    for row in r:
        try:
            vec = (float(row['x']), float(row['y']), float(row['z']))
        except:
            continue
        pix = direction_to_pixel(vec, cam)
        if pix is not None:
            u, v = pix
            w.writerow([row.get('id', ''), row.get('ra_deg', ''), row.get('dec_deg', ''), row.get('mag', ''),
                        row['x'], row['y'], row['z'], f"{u:.3f}", f"{v:.3f}"])
print("Wrote visible projected stars to:", out)
print("Cam params:", cam)
