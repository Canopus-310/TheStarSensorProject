# src/vision/catalog_based_generator.py
"""
Photorealistic catalog-based starfield generator (PATCHED).

Key change (Fix A): deposit flux as a subpixel-centered PSF for each star
instead of rounding to a single integer pixel then convolving. This preserves
subpixel information, spreads flux over multiple pixels so Poisson sampling
becomes meaningful for faint stars, and avoids piling flux into a handful of pixels.

Outputs:
 - results/catalog_sky_01_linear16.png  (uint16 linear sensor image)
 - results/catalog_sky_01_tonemapped.png (8-bit tone-mapped for display)
 - results/catalog_sky_01_overlay.png (tonemapped with truth markers)
 - results/catalog_sky_01_truth.csv (id, ra_deg, dec_deg, u, v, mag, electrons_deposited)
"""
import os, csv, math
import numpy as np
import cv2

# -------- USER-TUNABLE PARAMETERS ----------
INPUT_PROJECTED = "C:/Users/aryan/PycharmProjects/TheStarSensorProject/results/catalog_projected.csv"
OUT_TRUTH = "C:/Users/aryan/PycharmProjects/TheStarSensorProject/results/catalog_sky_01_truth.csv"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # project root
OUT_DIR = os.path.join(ROOT, "results")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_LINEAR16 = os.path.join(OUT_DIR, "catalog_sky_01_linear16.png")
OUT_TONEMAP  = os.path.join(OUT_DIR, "catalog_sky_01_tonemapped.png")
OUT_OVERLAY  = os.path.join(OUT_DIR, "catalog_sky_01_overlay.png")

WIDTH = 640
HEIGHT = 480

# PSF: sigma in pixels (optics)
PSF_SIGMA = 1.0


# Photometric mapping: convert magnitude -> mean photons/sec (relative)
MAG0 = 0.0            # reference magnitude
PHOTONS_OF_MAG0 = 5000   # 10× lower → faint stars survive

EXPOSURE = 1.0           # or set to 2.0 to increase further


# Noise model
READ_NOISE_ELECTRONS = 6.0   # one-sigma read noise (e-)
QUANT_MAX_UINT16 = 65535

# Tone-mapping params (for display only)
PERCENTILE_FOR_EXPOSURE = 99.5
DISPLAY_HEADROOM = 0.25   # how much of 65535 mapped to white before gamma
GAMMA = 1/2.2              # display gamma (linear->srgb-ish)
# -------------------------------------------

def mag_to_electrons(mag, mag0=MAG0, photons_mag0=PHOTONS_OF_MAG0, exposure=EXPOSURE):
    # linear electrons for this exposure (photometric: 10^(-0.4*(m-m0)))
    return photons_mag0 * (10.0 ** (-0.4 * (mag - mag0))) * exposure

# --- Read projected catalog ---
rows = []
if not os.path.exists(INPUT_PROJECTED):
    raise SystemExit(f"Missing input projected catalog: {INPUT_PROJECTED}")
with open(INPUT_PROJECTED, newline='', encoding='utf-8') as f:
    r = csv.DictReader(f)
    for row in r:
        try:
            u = float(row['u']); v = float(row['v'])
            mag = float(row.get('mag', 15.0))
        except Exception:
            continue
        rows.append({'row': row, 'u': u, 'v': v, 'mag': mag})

print(f"Loaded projected catalog rows: {len(rows)}")

# --- Prepare image (float electrons expected per pixel) ---
img = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
truth_rows = []

# Precompute PSF grid parameters
sigma = PSF_SIGMA
ksize = max(3, int(6.0 * sigma) | 1)  # odd kernel size
half = ksize // 2

# Precompute a coordinate grid for PSF centered at (0,0) integer grid indices
# But we'll compute per-star kernel using subpixel offsets (rx, ry) below.
y_idx, x_idx = np.mgrid[-half:half+1, -half:half+1].astype(np.float64)

# Loop over stars and deposit PSF per star at subpixel offset
for e in rows:
    u, v, mag = e['u'], e['v'], e['mag']
    electrons_total = mag_to_electrons(mag)

    # Skip stars obviously outside sensor bounds (with small margin for PSF)
    if (u < -2*ksize) or (u > WIDTH + 2*ksize) or (v < -2*ksize) or (v > HEIGHT + 2*ksize):
        continue

    # Build PSF centered at exact float location (u, v)
    # Pixel centers at integer coordinates; kernel sample coords relative to star center:
    # rx = x_coord_of_kernel - u, ry = y_coord_of_kernel - v
    # Choose integer patch anchors
    cx = int(np.floor(u))
    cy = int(np.floor(v))

    # Create kernel coordinates corresponding to image pixels [cy-half:cy+half, cx-half:cx+half]
    xx = (x_idx + cx)  # x coordinate for each kernel cell
    yy = (y_idx + cy)  # y coordinate for each kernel cell

    # relative coords from star center (float)
    rx = xx - u
    ry = yy - v

    kern = np.exp(-(rx**2 + ry**2) / (2.0 * sigma * sigma))
    s = kern.sum()
    if s <= 0:
        continue
    kern /= s  # normalized PSF

    # Determine overlap with image bounds
    x0_img = max(0, cx - half)
    x1_img = min(WIDTH, cx + half + 1)
    y0_img = max(0, cy - half)
    y1_img = min(HEIGHT, cy + half + 1)

    kx0 = x0_img - (cx - half)
    kx1 = kx0 + (x1_img - x0_img)
    ky0 = y0_img - (cy - half)
    ky1 = ky0 + (y1_img - y0_img)

    if kx1 <= kx0 or ky1 <= ky0:
        continue

    # electrons apportioned to this patch
    patch = kern[ky0:ky1, kx0:kx1] * electrons_total

    # Add into img (expected electrons per pixel)
    img[y0_img:y1_img, x0_img:x1_img] += patch

    # record how many electrons were deposited (sum may be slightly less at edges)
    electrons_deposited = float(patch.sum())
    truth_rows.append([
        e['row'].get('id', ''),
        e['row'].get('ra_deg', ''),
        e['row'].get('dec_deg', ''),
        f"{u:.3f}",
        f"{v:.3f}",
        f"{mag:.3f}",
        f"{electrons_deposited:.6f}"
    ])

# --- At this point img is expected electrons per pixel (float) ---
# Quick sanity
max_expected = float(img.max()) if img.size else 0.0
print("Max expected electrons per pixel (before noise):", max_expected)
print("Stars in truth file (rows):", len(truth_rows))

# --- Simulate Poisson photon noise (on electrons) ---
rng = np.random.default_rng()
# Poisson expects lambda >= 0; for very small lambdas it often returns 0 -> acceptable physically
img_poisson = rng.poisson(img.clip(min=0.0)).astype(np.float64)

# Add read noise (Gaussian) in electrons
img_noisy = img_poisson + rng.normal(loc=0.0, scale=READ_NOISE_ELECTRONS, size=img_poisson.shape)

# Clip to non-negative
img_noisy = np.clip(img_noisy, 0.0, None)

# --- Save truth file (electrons deposited BEFORE noise) ---
with open(OUT_TRUTH, "w", newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(["id","ra_deg","dec_deg","u","v","mag","electrons_deposited"])
    w.writerows(truth_rows)
print("Wrote truth file:", OUT_TRUTH)

# ---- Save linear 16-bit image (raw sensor-like) ----
# Robust linear scaling: map a high percentile to top of uint16 instead of absolute max
p_top = 99.9  # percentile to use for scaling; tweak 99.5..99.99
top_val = float(np.percentile(img_noisy, p_top))
if top_val <= 0:
    top_val = img_noisy.max() if img_noisy.max() > 0 else 1.0
scale_lin = QUANT_MAX_UINT16 / top_val
# After scaling we still clip to uint16 range. This prevents single extreme pixels dominating.

img_lin16 = np.clip((img_noisy * scale_lin).round(), 0, QUANT_MAX_UINT16).astype(np.uint16)
cv2.imwrite(OUT_LINEAR16, img_lin16)
print("Saved linear 16-bit:", OUT_LINEAR16)

# ---- Tone-map for display (8-bit) ----
# Use percentile-based exposure: map percentile -> DISPLAY_HEADROOM * max_uint16
p = np.percentile(img_noisy, PERCENTILE_FOR_EXPOSURE)
if p <= 0:
    p = max_e
display_scale = (DISPLAY_HEADROOM * QUANT_MAX_UINT16) / p
img_disp = img_noisy * display_scale

# clamp and optionally use a soft/log compression to reveal faint stars without blowing out bright ones
img_disp = np.clip(img_disp / float(QUANT_MAX_UINT16), 0.0, 1.0)
# small log compression to compress dynamic range (helps with many magnitudes)
img_disp = np.log1p(img_disp * 100.0) / np.log1p(100.0)  # tunable factor 100
# gamma for appearance
img_disp = np.power(np.clip(img_disp, 0.0, 1.0), GAMMA)
img_u8 = (img_disp * 255.0).round().astype(np.uint8)
cv2.imwrite(OUT_TONEMAP, img_u8)
print("Saved tone-mapped 8-bit:", OUT_TONEMAP)

# ---- Overlay truth markers on tone-mapped image ----
img_vis = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
for r in truth_rows:
    u = int(round(float(r[3]))); v = int(round(float(r[4])))
    if 0 <= u < WIDTH and 0 <= v < HEIGHT:
        cv2.circle(img_vis, (u, v), 2, (0,0,255), 1)
cv2.imwrite(OUT_OVERLAY, img_vis)
print("Saved overlay (truth markers):", OUT_OVERLAY)

# Summary
nstars_visible_pixels = np.count_nonzero(img_noisy > 0)
print("Nonzero electrons pixels (after noise):", nstars_visible_pixels)
