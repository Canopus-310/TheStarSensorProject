# src/scripts/project_catalog_to_pixels_point_at_bright.py
# Deterministic projector with strict angular clipping + debug prints
import csv, math, numpy as np, os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INFILE  = os.path.join(ROOT, "data", "catalog_unit_vectors.csv")
OUTFILE = os.path.join(ROOT, "results", "catalog_projected.csv")

# ---------- USER TUNE ----------
WIDTH  = 640
HEIGHT = 480
FOV_X_DEG = 0.0001   # <-- set this to what you want (degrees)
# -------------------------------

# Derived
half_fov_deg = float(FOV_X_DEG) / 2.0
half_fov_cos = math.cos(math.radians(half_fov_deg))

print(f"[projector] WIDTH={WIDTH} HEIGHT={HEIGHT} FOV_X_DEG={FOV_X_DEG} half_fov_deg={half_fov_deg}")

# Camera boresight (inertial)
TARGET = np.array([1.0, 0.0, 0.0])
cam_dir = TARGET / np.linalg.norm(TARGET)

# Build right/up basis for camera (not strictly needed for angle test, but for projection)
WORLD_UP = np.array([0.0, 0.0, 1.0])
if abs(np.dot(cam_dir, WORLD_UP)) > 0.99:
    WORLD_UP = np.array([0.0, 1.0, 0.0])
right = np.cross(cam_dir, WORLD_UP)
rn = np.linalg.norm(right)
if rn < 1e-12: right = np.array([0.0, 1.0, 0.0])
else: right = right / rn
up = np.cross(right, cam_dir); up = up / np.linalg.norm(up)

# pinhole intrinsics (square pixels)
FOV_X = math.radians(FOV_X_DEG)
fx = (WIDTH / 2.0) / math.tan(FOV_X / 2.0)
fy = fx
cx = WIDTH / 2.0
cy = HEIGHT / 2.0

def project_unit_vector(v_in):
    z_cam = float(np.dot(v_in, cam_dir))
    if z_cam <= 0.0:
        return None
    x_cam = float(np.dot(v_in, right))
    y_cam = float(np.dot(v_in, up))
    xn = x_cam / z_cam
    yn = y_cam / z_cam
    u = fx * xn + cx
    v = -fy * yn + cy
    if 0.0 <= u < WIDTH and 0.0 <= v < HEIGHT:
        return (u, v)
    return None

# remove previous file to avoid confusion
try:
    if os.path.exists(OUTFILE):
        os.remove(OUTFILE)
except Exception:
    pass

rows = []
skipped = 0
ang_debug = []

with open(INFILE, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for i, row in enumerate(r):
        try:
            x = float(row["x"]); y = float(row["y"]); z = float(row["z"])
        except Exception:
            continue
        v = np.array([x, y, z])
        # exact angular separation (in degrees) between v and cam_dir
        cosang = np.dot(v, cam_dir)
        # numerical safety
        cosang = max(-1.0, min(1.0, float(cosang)))
        ang_deg = math.degrees(math.acos(cosang))
        # store some debug samples (first few)
        if len(ang_debug) < 6:
            ang_debug.append((i, ang_deg, cosang))
        # strict clipping
        if ang_deg <= half_fov_deg:
            proj = project_unit_vector(v)
            if proj is None:
                # rare: projected out of pixel bounds
                skipped += 1
                continue
            u, v_pix = proj
            rows.append([ row.get("id",""), row.get("ra_deg",""), row.get("dec_deg",""),
                          row.get("mag","9.0"),
                          f"{x:.12g}", f"{y:.12g}", f"{z:.12g}",
                          f"{u:.3f}", f"{v_pix:.3f}" ])
        else:
            skipped += 1

# write out
os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)
with open(OUTFILE, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id","ra_deg","dec_deg","mag","x","y","z","u","v"])
    w.writerows(rows)

print(f"[projector] Projected rows: {len(rows)}")
print(f"[projector] Skipped rows: {skipped}")
print("[projector] Sample angular debug (index, deg, cos):")
for t in ang_debug:
    print("   ", t)
print("[projector] Wrote:", OUTFILE)
