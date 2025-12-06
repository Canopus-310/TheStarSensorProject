# src/vision/catalog_based_generator.py
"""
Photorealistic catalog-based starfield generator.

Outputs:
 - results/catalog_sky_01_linear16.png  (uint16 linear sensor image)
 - results/catalog_sky_01_tonemapped.png (8-bit tone-mapped for display)
 - results/catalog_sky_01_overlay.png (tonemapped with truth markers)
 - results/catalog_sky_01_truth.csv (id, ra_deg, dec_deg, u, v, mag, electrons_deposited)

Tweak PARAMETERS below for realism vs visibility.
"""
import os, csv, math, numpy as np, cv2

# -------- USER-TUNABLE PARAMETERS ----------
INPUT_PROJECTED = "C:/Users/aryan/PycharmProjects/TheStarSensorProject/results/catalog_projected.csv"
OUT_TRUTH = "C:/Users/aryan/PycharmProjects/TheStarSensorProject/results/catalog_sky_01_truth.csv"
OUT_LINEAR16 = "C:/Users/aryan\PycharmProjects\TheStarSensorProject/results/catalog_sky_01_linear16.png"
OUT_TONEMAP = "C:/Users/aryan\PycharmProjects\TheStarSensorProject/results/catalog_sky_01_tonemapped.png"
OUT_OVERLAY = "C:/Users/aryan\PycharmProjects\TheStarSensorProject/results/catalog_sky_01_overlay.png"

WIDTH = 640
HEIGHT = 480

# PSF: sigma in pixels (optics)
PSF_SIGMA = 1.6

# Photometric mapping: convert magnitude -> mean photons/sec (relative)
# We'll map magnitudes to electrons collected in an exposure time.
MAG0 = 0.0            # reference magnitude
PHOTONS_OF_MAG0 = 5e4 # electrons for magnitude MAG0 (set scale - adjust for realism)
EXPOSURE = 1.0        # seconds

# Noise model
READ_NOISE_ELECTRONS = 6.0   # one-sigma read noise (e-)
QUANT_MAX_UINT16 = 65535

# Tone-mapping params (for display only)
PERCENTILE_FOR_EXPOSURE = 99.5
DISPLAY_HEADROOM = 0.25   # how much of 65535 mapped to white before gamma
GAMMA = 1/2.2              # display gamma (linear->srgb-ish)
# -------------------------------------------

os.makedirs("results", exist_ok=True)

def mag_to_electrons(mag, mag0=MAG0, photons_mag0=PHOTONS_OF_MAG0, exposure=EXPOSURE):
    # linear electrons for this exposure (photometric: 10^(-0.4*(m-m0)))
    return photons_mag0 * (10.0 ** (-0.4 * (mag - mag0))) * exposure

# read projected catalog
rows = []
with open(INPUT_PROJECTED, newline='', encoding='utf-8') as f:
    r = csv.DictReader(f)
    for row in r:
        try:
            u = float(row['u']); v = float(row['v'])
            mag = float(row.get('mag', 15.0))
        except Exception:
            continue
        # store original catalog coords too if present
        rows.append({'row': row, 'u': u, 'v': v, 'mag': mag})

# deposit electrons (float image)
img = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
truth_rows = []
for e in rows:
    u, v, mag = e['u'], e['v'], e['mag']
    # deposit at nearest integer pixel as delta (will be convolved)
    iu = int(round(u))
    iv = int(round(v))
    if 0 <= iu < WIDTH and 0 <= iv < HEIGHT:
        electrons = mag_to_electrons(mag)
        img[iv, iu] += electrons
        truth_rows.append([e['row'].get('id',''), e['row'].get('ra_deg',''), e['row'].get('dec_deg',''),
                           f"{u:.3f}", f"{v:.3f}", f"{mag:.3f}", f"{electrons:.6f}"])

# Convolve with Gaussian PSF (float64)
sigma = PSF_SIGMA
ksize = max(1, int(6 * sigma) | 1)
img_blurred = cv2.GaussianBlur(img.astype(np.float32), (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE).astype(np.float64)

# Simulate Poisson photon noise (on electrons)
# To keep distributions realistic, we treat img_blurred as expected electrons per exposure
rng = np.random.default_rng()
img_poisson = rng.poisson(img_blurred.clip(min=0.0)).astype(np.float64)

# Add read noise (Gaussian) in electrons
img_noisy = img_poisson + rng.normal(loc=0.0, scale=READ_NOISE_ELECTRONS, size=img_poisson.shape)

# Clip to non-negative
img_noisy = np.clip(img_noisy, 0.0, None)

# Save truth file (electrons deposited BEFORE noise, for reference we use img_blurred peak per star)
with open(OUT_TRUTH, "w", newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(["id","ra_deg","dec_deg","u","v","mag","electrons_deposited"])
    w.writerows(truth_rows)

# ---- Save linear 16-bit image (realistic raw look) ----
# scale such that max maps to QUANT_MAX_UINT16
max_e = img_noisy.max() if img_noisy.max() > 0 else 1.0
scale_lin = QUANT_MAX_UINT16 / max_e
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
# normalize to [0,1]
img_disp = img_disp / float(QUANT_MAX_UINT16)
img_disp = np.clip(img_disp, 0.0, 1.0)
# gamma
img_disp = np.power(img_disp, GAMMA)
img_u8 = (img_disp * 255.0).round().astype(np.uint8)
cv2.imwrite(OUT_TONEMAP, img_u8)
print("Saved tone-mapped 8-bit:", OUT_TONEMAP)

# ---- Overlay truth markers on tone-mapped image ----
img_vis = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
for r in truth_rows:
    u = int(round(float(r[3]))); v = int(round(float(r[4])))
    if 0 <= u < WIDTH and 0 <= v < HEIGHT:
        cv2.circle(img_vis, (u,v), 2, (0,0,255), 1)
cv2.imwrite(OUT_OVERLAY, img_vis)
print("Saved overlay (truth markers):", OUT_OVERLAY)

# Summary
nstars_visible = np.count_nonzero(img_blurred > 0)
print("Stars in truth file (rows):", len(truth_rows))
print("Nonzero blurred pixels:", nstars_visible)
