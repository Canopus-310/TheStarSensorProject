# tools/filter_and_match.py
import os, csv, subprocess, sys, math
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DET_IN = os.path.join(ROOT, "results_archived_2", "detected_centroids.csv")
# if you previously used shifted file, change this to detected_centroids_shifted.csv
if not os.path.exists(DET_IN):
    DET_IN = os.path.join(ROOT, "results_archived_2", "detected_centroids_shifted.csv")
DET_OUT = os.path.join(ROOT, "results_archived_2", "detected_centroids_filtered.csv")
TRUTH = os.path.join(ROOT, "results_archived_2", "catalog_sky_01_truth.csv")
IMG_PATH = os.path.join(ROOT, "results_archived_2", "catalog_sky_01_linear16.png")

W = 640
H = 480
# keep only detections inside image bounds and within center radius
KEEP_RADIUS = 150.0  # px; tweak smaller/larger if needed

# compute image center
cx = W / 2.0
cy = H / 2.0

rows = []
hdr = None
with open(DET_IN, newline='', encoding='utf-8') as f:
    r = csv.DictReader(f)
    hdr = r.fieldnames
    for row in r:
        try:
            x = float(row['x']); y = float(row['y'])
        except Exception:
            continue
        # filter bounds
        if not (0.0 <= x < W and 0.0 <= y < H):
            continue
        # filter central radius
        if math.hypot(x - cx, y - cy) > KEEP_RADIUS:
            continue
        rows.append((x, y, row))

print(f"Loaded {len(rows)} detections after filtering (radius {KEEP_RADIUS}px)")

# write filtered csv (preserve header)
with open(DET_OUT, "w", newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(hdr)
    for (x,y,row) in rows:
        out = [row.get(h, "") for h in hdr]
        xi = hdr.index('x')
        yi = hdr.index('y')
        out[xi] = f"{x:.6f}"
        out[yi] = f"{y:.6f}"
        w.writerow(out)

print("Wrote filtered detections to:", DET_OUT)

# call matcher
cmd = [
    sys.executable,
    os.path.join(ROOT, "src", "scripts", "match_detections_to_truth.py"),
    "--det", DET_OUT,
    "--truth", TRUTH,
    "--out", os.path.join(ROOT, "results_archived_2", "match_results_filtered.csv"),
    "--thresh", "8"
]
print("Running matcher:", " ".join(cmd))
proc = subprocess.run(cmd, capture_output=True, text=True)
print(proc.stdout)
print(proc.stderr)
print("Matcher exit code:", proc.returncode)
