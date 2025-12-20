# tools/inspect_match.py
import csv, os, math
import numpy as np
import cv2
from scipy.spatial import cKDTree

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DET_PATH = os.path.join(ROOT, "results_archived_2", "detected_centroids.csv")
TRUTH_PATH = os.path.join(ROOT, "results_archived_2", "catalog_sky_01_truth.csv")
IMG_PATH = os.path.join(ROOT, "results_archived_2", "catalog_sky_01_linear16.png")
OUT_DIR = os.path.join(ROOT, "results_archived_2", "debug_inspect")
os.makedirs(OUT_DIR, exist_ok=True)

THRESH = 8.0  # matching threshold in pixels
# load detections
dets = []
with open(DET_PATH, newline='', encoding='utf-8') as f:
    r = csv.DictReader(f)
    for row in r:
        try:
            x = float(row['x']); y = float(row['y'])
            dets.append((x,y))
        except Exception:
            continue
dets = np.array(dets)
# load truth
truth = []
with open(TRUTH_PATH, newline='', encoding='utf-8') as f:
    r = csv.DictReader(f)
    for row in r:
        try:
            u = float(row['u']); v = float(row['v'])
            truth.append((u,v))
        except Exception:
            continue
truth = np.array(truth)

print("Loaded detections:", dets.shape)
print("Loaded truth rows :", truth.shape)

# quick stats
def stats(a, name):
    if a.size == 0:
        print(name, "EMPTY")
        return
    print(f"{name}: minx={a[:,0].min():.3f} maxx={a[:,0].max():.3f} miny={a[:,1].min():.3f} maxy={a[:,1].max():.3f} meanx={a[:,0].mean():.3f} meany={a[:,1].mean():.3f}")

stats(dets, "DETECTIONS")
stats(truth, "TRUTH")

if dets.size == 0 or truth.size == 0:
    raise SystemExit("Nothing to compare.")

img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise SystemExit("Could not load image: " + IMG_PATH)
h,w = img.shape[:2]
print("Image size:", w, "x", h)

# utility to test a variant
def match_variant(dets_arr, truth_arr, name):
    tree = cKDTree(truth_arr)
    dists, idxs = tree.query(dets_arr, k=1)
    matched = (dists <= THRESH).sum()
    mean = float(np.mean(dists))
    med = float(np.median(dists))
    mx = float(np.max(dists))
    print(f"[{name}] matched={matched}/{len(dets)} mean={mean:.2f} med={med:.2f} max={mx:.2f}")
    return matched, dists, idxs

variants = {}
variants['as_is'] = (dets.copy(), truth.copy())
# try swapping det x/y
variants['swap_det'] = (dets[:, ::-1].copy(), truth.copy())
# try swapping truth x/y
variants['swap_truth'] = (dets.copy(), truth[:, ::-1].copy())
# try swapping both sides
variants['swap_both'] = (dets[:, ::-1].copy(), truth[:, ::-1].copy())
# invert u/v axes as earlier
variants['invert_v'] = (dets.copy(), np.column_stack([truth[:,0], h - truth[:,1]]))
variants['invert_u'] = (dets.copy(), np.column_stack([w - truth[:,0], truth[:,1]]))
variants['invert_both'] = (dets.copy(), np.column_stack([w - truth[:,0], h - truth[:,1]]))
# combine swap+invert
variants['swap_det_invert_truth'] = (dets[:, ::-1].copy(), np.column_stack([w - truth[:,0], truth[:,1]]))
variants['swap_both_invert'] = (dets[:, ::-1].copy(), np.column_stack([w - truth[:,0], h - truth[:,1]]))

results = {}
for name, (darr, tarr) in variants.items():
    try:
        results[name] = match_variant(darr, tarr, name)
    except Exception as e:
        print("Error for", name, e)

# pick best by matched count
best = max(results.items(), key=lambda x: x[1][0])
print("BEST variant:", best[0], "with", best[1][0], "matches")

# save overlay image for best variant
best_name = best[0]
darr, tarr = variants[best_name]
vis = cv2.cvtColor((img / max(1,img.max()) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
for (u,v) in tarr[:5000]:
    cv2.circle(vis, (int(round(u)), int(round(v))), 1, (0,255,0), -1)
for (x,y) in darr:
    cv2.circle(vis, (int(round(x)), int(round(y))), 2, (0,0,255), 1)
outf = os.path.join(OUT_DIR, f"overlay_best_{best_name}.png")
cv2.imwrite(outf, vis)
print("Wrote overlay for best variant to:", outf)

# Also write per-detection nearest-distances (for further exploration)
tree_best = cKDTree(tarr)
dists, idxs = tree_best.query(darr, k=1)
np.savetxt(os.path.join(OUT_DIR, "dists_best_variant.csv"), np.column_stack([darr[:,0], darr[:,1], tarr[idxs][:,0], tarr[idxs][:,1], dists]), delimiter=",", header="det_x,det_y,truth_x,truth_y,dist", comments="")
print("Saved dists CSV to results_archived_2/debug_inspect/dists_best_variant.csv")
