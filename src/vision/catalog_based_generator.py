# src/vision/catalog_based_generator.py
import csv, numpy as np, cv2, os, argparse
from pathlib import Path
from scipy.ndimage import gaussian_filter

# ------------------------
# CLI for output dir
# ------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--projected_catalog", required=True,
                help="Path to catalog_projected.csv")
ap.add_argument("--out_dir", required=True,
                help="Directory to write synthetic images + truth")
ap.add_argument("--centroid-noise", type=float, default=0.0)
args = ap.parse_args()

OUT_DIR = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

projected_path = Path(args.projected_catalog)
if not projected_path.exists():
    raise RuntimeError(f"Projected catalog not found: {projected_path}")

# Outputs (ALL inside experiment directory)
truth_path   = OUT_DIR / "catalog_sky_01_truth.csv"
img16_path   = OUT_DIR / "catalog_sky_01_linear16.png"
img8_path    = OUT_DIR / "catalog_sky_01_tonemapped.png"
overlay_path = OUT_DIR / "catalog_sky_01_overlay.png"

# -----------------------------------------------------
# Load projected catalog
# -----------------------------------------------------
rows = []
with open(projected_path, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        try:
            u = float(row["u"]); v = float(row["v"])
            mag = float(row["mag"])
            x = float(row["x"]); y = float(row["y"]); z = float(row["z"])
        except:
            continue
        rows.append([u, v, mag, x, y, z])

print(f"Loaded projected catalog rows: {len(rows)}")

# -----------------------------------------------------
# Image generation
# -----------------------------------------------------
HEIGHT = 480
WIDTH  = 640

img = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
truth_out = []

max_e = 0.0

centroid_noise = float(args.centroid_noise)

for u, v, mag, x, y, z in rows:

    # ------------------------------------------------
    # NEW: Optional centroid noise injection
    # ------------------------------------------------
    if centroid_noise > 0:
        u += np.random.normal(0, centroid_noise)
        v += np.random.normal(0, centroid_noise)

    # brightness model
    electrons = 10_000 / (mag + 1.0)
    max_e = max(max_e, electrons)

    ui = int(round(u))
    vi = int(round(v))

    if 0 <= ui < WIDTH and 0 <= vi < HEIGHT:
        img[vi, ui] += electrons
        truth_out.append([u, v, mag, electrons])

# -----------------------------------------------------
# Apply optical PSF (Gaussian blur)
# -----------------------------------------------------
PSF_SIGMA_PX = 1.0   # this is realistic for small optics
img = gaussian_filter(img, sigma=PSF_SIGMA_PX)

print("Max expected electrons per pixel (before noise):", max_e)
print("Stars in truth file (rows):", len(truth_out))

# -----------------------------------------------------
# Save truth CSV
# -----------------------------------------------------
with open(truth_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["u","v","mag","electrons"])
    w.writerows(truth_out)
print("Wrote truth file:", truth_path)



# -----------------------------------------------------
# Noise model
# -----------------------------------------------------
noise_img = img + np.random.normal(0, 1.0, img.shape)
noise_img = np.clip(noise_img, 0, None)

# Save 16-bit
img16 = np.clip(noise_img, 0, 65535).astype(np.uint16)
cv2.imwrite(str(img16_path), img16)
print("Saved linear 16-bit:", img16_path)

# Save 8-bit for visualization
img8 = np.clip((noise_img / noise_img.max()) * 255, 0, 255).astype(np.uint8)
cv2.imwrite(str(img8_path), img8)
print("Saved tone-mapped 8-bit:", img8_path)

# Overlay truth stars
img_overlay = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
for (u, v, mag, e) in truth_out:
    cv2.circle(img_overlay, (int(u), int(v)), 1, (0,0,255), 1)

cv2.imwrite(str(overlay_path), img_overlay)
print("Saved overlay (truth markers):", overlay_path)

print("Nonzero electrons pixels (after noise):", int((img16 > 0).sum()))
