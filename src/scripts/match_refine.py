# src/scripts/match_refine.py
import csv
import argparse
from pathlib import Path

# ---------------- CLI ----------------
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max_err_px", type=float, default=5.0)
args = ap.parse_args()

INFILE = Path(args.input)
OUTFILE = Path(args.out)

rows = []
seen_truth = set()

with open(INFILE, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        dist = float(row["dist"])
        truth_idx = row["truth_idx"]

        # reject bad matches
        if dist > args.max_err_px:
            continue

        # enforce 1-to-1 truth correspondence
        if truth_idx in seen_truth:
            continue

        seen_truth.add(truth_idx)
        rows.append(row)

if not rows:
    raise RuntimeError("No refined matches survived")

with open(OUTFILE, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print(f"[refine] input rows     = {len(seen_truth)}")
print(f"[refine] refined unique = {len(rows)}")
print(f"[refine] wrote {OUTFILE}")
