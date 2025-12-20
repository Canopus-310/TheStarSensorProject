# tools/shift_dets_and_match.py
import csv, os, subprocess, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DET_IN = os.path.join(ROOT, "results_archived_2", "detected_centroids.csv")
TRUTH = os.path.join(ROOT, "results_archived_2", "catalog_sky_01_truth.csv")
DET_OUT = os.path.join(ROOT, "results_archived_2", "detected_centroids_shifted.csv")

# load detections
dets = []
with open(DET_IN, newline='', encoding='utf-8') as f:
    r = csv.DictReader(f)
    det_hdr = r.fieldnames
    for row in r:
        try:
            x = float(row['x']); y = float(row['y'])
            dets.append((x,y,row))
        except Exception:
            continue
if not dets:
    print("No detections loaded from", DET_IN); sys.exit(1)

# load truth, compute mean
truth_xy = []
with open(TRUTH, newline='', encoding='utf-8') as f:
    r = csv.DictReader(f)
    for row in r:
        try:
            u = float(row['u']); v = float(row['v'])
            truth_xy.append((u,v))
        except Exception:
            continue
if not truth_xy:
    print("No truth rows loaded from", TRUTH); sys.exit(1)

dets_xy = np.array([[d[0], d[1]] for d in dets])
truth_xy = np.array(truth_xy)

mean_det = dets_xy.mean(axis=0)
mean_truth = truth_xy.mean(axis=0)
dx = mean_truth[0] - mean_det[0]
dy = mean_truth[1] - mean_det[1]

print("mean_det:", mean_det, "mean_truth:", mean_truth)
print("Applying shift dx,dy =", dx, dy)

# write shifted CSV (preserve header/order)
with open(DET_OUT, "w", newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    # write header - keep same as input (x,y,area,peak,snr etc.)
    w.writerow(det_hdr)
    for (x,y,row) in dets:
        newx = x + dx
        newy = y + dy
        # build row in same order as det_hdr
        out_row = [ row.get(h, "") for h in det_hdr ]
        # replace x,y fields
        # find indices
        xi = det_hdr.index('x')
        yi = det_hdr.index('y')
        out_row[xi] = f"{newx:.6f}"
        out_row[yi] = f"{newy:.6f}"
        w.writerow(out_row)

print("Wrote shifted detections to:", DET_OUT)
print("Now invoking matcher with shifted file...")

# run matcher (same as you used)
cmd = [
    sys.executable,
    os.path.join(ROOT, "src", "scripts", "match_detections_to_truth.py"),
    "--det", DET_OUT,
    "--truth", TRUTH,
    "--out", os.path.join(ROOT, "results_archived_2", "match_results_shifted.csv"),
    "--thresh", "8"
]
print("Running:", " ".join(cmd))
proc = subprocess.run(cmd, capture_output=True, text=True)
print(proc.stdout)
print(proc.stderr)
print("Matcher exit code:", proc.returncode)
print("If matches improved, we should make this shift permanent in detector.py.")
