# scripts/match_detections_to_truth.py
"""
Usage:
  python scripts/match_detections_to_truth.py --det results_archived_2/detected_centroids.csv --truth results_archived_2/catalog_sky_01_truth.csv --det_cols x y --truth_cols u v --out results_archived_2/match_report.csv --thresh 6.0

Columns:
 - detector file should have two numeric columns for coordinates (defaults: x,y)
 - truth file should have columns (defaults: u,v)
"""
import csv, argparse, math, numpy as np, os

parser = argparse.ArgumentParser()
parser.add_argument("--det", required=True, help="detected centroids CSV")
parser.add_argument("--truth", required=True, help="truth CSV")
parser.add_argument("--det_cols", nargs=2, default=["x","y"], help="detected columns names")
parser.add_argument("--truth_cols", nargs=2, default=["u","v"], help="truth columns names")
parser.add_argument("--out", default="results_archived_2/match_report.csv")
parser.add_argument("--thresh", type=float, default=6.0, help="max pixels to consider a match")
args = parser.parse_args()

def load_points(fn, colx, coly):
    pts = []
    with open(fn, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                x = float(row[colx]); y = float(row[coly])
                pts.append((x,y,row))
            except Exception:
                continue
    return pts

det_pts = load_points(args.det, args.det_cols[0], args.det_cols[1])
truth_pts = load_points(args.truth, args.truth_cols[0], args.truth_cols[1])

if len(truth_pts) == 0:
    print("No truth points found in", args.truth); raise SystemExit(1)

# Build arrays
det_coords = np.array([[p[0], p[1]] for p in det_pts])
truth_coords = np.array([[p[0], p[1]] for p in truth_pts])

# For each detection, find nearest truth (brute force)
matches = []   # (det_idx, truth_idx, dist)
used_truth = set()
for i, d in enumerate(det_coords):
    if truth_coords.shape[0] == 0:
        break
    diffs = truth_coords - d
    dists = np.hypot(diffs[:,0], diffs[:,1])
    j = int(np.argmin(dists))
    matches.append((i, j, float(dists[j])))

# Now analyze: choose unique truth -> best match per truth
# For simplicity: for each truth pick closest detection (if within thresh)
truth_to_best = {}
for det_i, truth_i, dist in matches:
    if dist <= args.thresh:
        if truth_i not in truth_to_best or dist < truth_to_best[truth_i][0]:
            truth_to_best[truth_i] = (dist, det_i)

matched_dets = set([v[1] for v in truth_to_best.values()])
matched_truths = set(truth_to_best.keys())

# Stats
errors = [truth_to_best[t][0] for t in matched_truths]
mean_err = float(np.mean(errors)) if errors else float('nan')
median_err = float(np.median(errors)) if errors else float('nan')
max_err = float(np.max(errors)) if errors else float('nan')

false_pos = [i for i in range(len(det_pts)) if i not in matched_dets]
false_neg = [i for i in range(len(truth_pts)) if i not in matched_truths]

print("Detections:", len(det_pts))
print("Truth points:", len(truth_pts))
print("Matches (within thresh):", len(matched_truths))
print(f"Errors (px): mean={mean_err:.3f}, median={median_err:.3f}, max={max_err:.3f}")
print("False positives (detections unmatched):", len(false_pos))
print("False negatives (truth unmatched):", len(false_neg))

# write a small CSV of match pairs
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w", newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(["truth_idx","truth_x","truth_y","det_idx","det_x","det_y","dist"])
    for t_idx, (dist, det_idx) in truth_to_best.items():
        tx, ty = truth_coords[t_idx]
        dx, dy = det_coords[det_idx]
        w.writerow([t_idx, f"{tx:.3f}", f"{ty:.3f}", det_idx, f"{dx:.3f}", f"{dy:.3f}", f"{dist:.3f}"])
print("Wrote match CSV to", args.out)
