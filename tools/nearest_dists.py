# tools/nearest_dists.py
import csv, os, math
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DET = os.path.join(ROOT, "results", "detected_centroids_shifted.csv")
TRUTH = os.path.join(ROOT, "results", "catalog_sky_01_truth.csv")
OUT = os.path.join(ROOT, "results", "nearest_dists_debug.csv")

def load_xy(path, xkey='x', ykey='y'):
    pts = []
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                pts.append((float(row[xkey]), float(row[ykey])))
            except Exception:
                continue
    return np.array(pts, dtype=np.float64)

dets = load_xy(DET, xkey='x', ykey='y')
truth = load_xy(TRUTH, xkey='u', ykey='v')

print("Loaded:", DET, dets.shape, "and", TRUTH, truth.shape)
if dets.size == 0 or truth.size == 0:
    raise SystemExit("Empty detections or truth.")

# For each detection compute min distance to truth (brute force OK: 178 x 118k ~ 21M distances)
d_min = []
nearest_idx = []
for i, (dx, dy) in enumerate(dets):
    # squared distances vectorized
    ds2 = (truth[:,0] - dx)**2 + (truth[:,1] - dy)**2
    j = int(np.argmin(ds2))
    d_min.append(math.sqrt(float(ds2[j])))
    nearest_idx.append(j)
    if i < 10:
        print(f"det[{i}] = ({dx:.3f},{dy:.3f}) -> nearest truth[{j}] = ({truth[j,0]:.3f},{truth[j,1]:.3f}), dist = {math.sqrt(float(ds2[j])):.3f}")

d_min = np.array(d_min)
print("Nearest-dist stats (px): min={:.3f} mean={:.3f} med={:.3f} max={:.3f}".format(
    d_min.min(), d_min.mean(), np.median(d_min), d_min.max()))

# Save small CSV: det_x,det_y,truth_x,truth_y,dist
with open(OUT, "w", newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(["det_x","det_y","truth_x","truth_y","dist"])
    for (dx,dy), idx, dist in zip(dets, nearest_idx, d_min):
        w.writerow([f"{dx:.6f}", f"{dy:.6f}", f"{truth[idx,0]:.6f}", f"{truth[idx,1]:.6f}", f"{dist:.6f}"])
print("Wrote nearest results to:", OUT)
