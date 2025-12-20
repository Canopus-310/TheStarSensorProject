# tools/debug_match.py
import os, csv, math
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial import cKDTree

ROOT = Path("../..").resolve()
img_path = ROOT / "results_archived_2" / "catalog_sky_01_linear16.png"
det_path = ROOT / "results_archived_2" / "detected_centroids.csv"
truth_path = ROOT / "results_archived_2" / "catalog_sky_01_truth.csv"
out_dir = ROOT / "results_archived_2" / "debug_match"
out_dir.mkdir(parents=True, exist_ok=True)

THRESH = 8.0
PLOT_LIMIT = 5000

def load_detections(p):
    arr = []
    with open(p, newline='', encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                x = float(row["x"])
                y = float(row["y"])
                arr.append((x,y))
            except:
                continue
    return np.array(arr)

def load_truth(p):
    arr = []
    with open(p, newline='', encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                u = float(row["u"])
                v = float(row["v"])
                arr.append((u,v))
            except:
                continue
    return np.array(arr)

print("Loading image:", img_path)
img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
if img is None:
    raise SystemExit("COULD NOT LOAD IMAGE.")
h, w = img.shape[:2]
print("Image size:", w, "x", h)

dets = load_detections(det_path)
truth = load_truth(truth_path)

print("Loaded detections:", len(dets))
print("Loaded truth rows:", len(truth))

if len(dets) == 0:
    raise SystemExit("NO detections!")
if len(truth) == 0:
    raise SystemExit("NO truth stars!")

if PLOT_LIMIT and len(truth) > PLOT_LIMIT:
    truth_plot = truth[:PLOT_LIMIT]
else:
    truth_plot = truth

def analyze_variant(truth_pts, name):
    tree = cKDTree(truth_pts)
    # ------- FIX: remove n_jobs -------
    dists, idxs = tree.query(dets, k=1)
    # ----------------------------------

    matched = (dists <= THRESH).sum()
    mean = float(np.mean(dists))
    median = float(np.median(dists))
    maxd = float(np.max(dists))
    print(f"[{name}] matched={matched}/{len(dets)}, mean={mean:.2f}, med={median:.2f}, max={maxd:.2f}")

    vis = cv2.cvtColor((img / img.max() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    for (u,v) in truth_plot:
        cv2.circle(vis, (int(u), int(v)), 1, (0,255,0), -1)
    for (x,y) in dets:
        cv2.circle(vis, (int(x), int(y)), 2, (0,0,255), 1)

    cv2.imwrite(str(out_dir / f"overlay_{name}.png"), vis)

    outcsv = out_dir / f"dists_{name}.csv"
    with open(outcsv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["det_x","det_y","truth_x","truth_y","dist"])
        for (dx,dy), tidx, d in zip(dets, idxs, dists):
            tx, ty = truth_pts[tidx]
            w.writerow([dx, dy, tx, ty, d])

    return matched, mean, median, maxd

variants = {
    "as_is": truth.copy(),
    "invert_v": np.column_stack([truth[:,0], h - truth[:,1]]),
    "invert_u": np.column_stack([w - truth[:,0], truth[:,1]]),
    "invert_both": np.column_stack([w - truth[:,0], h - truth[:,1]]),
}

print("\nAnalyzing variants...")
results = {}

for name, pts in variants.items():
    results[name] = analyze_variant(pts, name)

best = max(results.items(), key=lambda x: x[1][0])
print("\nBest:", best[0])
print("All overlays & dists saved in:", out_dir)
