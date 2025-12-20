# tools/refine_centroids.py
import csv, math, os
import numpy as np
import cv2
from scipy.optimize import least_squares

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMG = os.path.join(ROOT, "results_archived_2", "catalog_sky_01_linear16.png")
DET_IN = os.path.join(ROOT, "results_archived_2", "detected_centroids.csv")
DET_OUT = os.path.join(ROOT, "results_archived_2", "detected_centroids_refined.csv")

# window size (odd). Use ~ (4 * psf_sigma). Try 9 or 11.
WIN = 9

def gaussian2d(params, x, y):
    A, x0, y0, sx, sy, b = params
    return A * np.exp(-(((x - x0)**2)/(2*sx*sx) + ((y - y0)**2)/(2*sy*sy))) + b

def residuals(params, patch, X, Y):
    return (gaussian2d(params, X, Y) - patch).ravel()

# load image
img = cv2.imread(IMG, -1)
if img is None:
    raise SystemExit(f"Could not open image: {IMG}")
img = img.astype(np.float64)
h, w = img.shape[:2]

# read detections
dets = []
with open(DET_IN, newline='', encoding='utf-8') as f:
    r = csv.DictReader(f)
    rows = list(r)

if len(rows) == 0:
    raise SystemExit("No detections found in " + DET_IN)

for row in rows:
    cx = float(row['x']); cy = float(row['y'])
    x0 = int(round(cx)); y0 = int(round(cy))
    x1 = max(0, x0 - WIN//2); x2 = min(w, x0 + WIN//2 + 1)
    y1 = max(0, y0 - WIN//2); y2 = min(h, y0 + WIN//2 + 1)
    patch = img[y1:y2, x1:x2].astype(np.float64)
    if patch.size == 0 or patch.shape[0] < 3 or patch.shape[1] < 3:
        dets.append((cx, cy))
        continue
    yy, xx = np.mgrid[y1:y2, x1:x2]
    # initial guess
    A0 = float(patch.max() - patch.min())
    if A0 <= 0:
        dets.append((cx, cy))
        continue
    p0 = [A0, cx, cy, 1.5, 1.5, float(patch.min())]
    try:
        res = least_squares(residuals, p0, args=(patch, xx, yy), max_nfev=200)
        A, xb, yb, sx, sy, b = res.x
        dets.append((xb, yb))
    except Exception:
        dets.append((cx, cy))

# write refined CSV (copy all fields, replace x,y)
with open(DET_OUT, "w", newline='', encoding='utf-8') as g:
    fieldnames = rows[0].keys()
    wout = csv.DictWriter(g, fieldnames=fieldnames)
    wout.writeheader()
    for i, row in enumerate(rows):
        row['x'] = f"{dets[i][0]:.6f}"
        row['y'] = f"{dets[i][1]:.6f}"
        wout.writerow(row)

print("Wrote refined detections:", DET_OUT)
