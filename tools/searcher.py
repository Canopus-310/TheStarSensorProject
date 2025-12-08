# tools/summarize_matches.py
import csv, math, os
fn = "results/match_results.csv"
if not os.path.exists(fn):
    print("Missing:", fn); raise SystemExit(1)
errs=[]
matched=0
with open(fn, newline='', encoding='utf-8') as f:
    r = csv.DictReader(f)
    for row in r:
        matched += 1
        try:
            errs.append(float(row['pixel_error']))
        except:
            pass
print("Matched rows:", matched)
if matched:
    errs_sorted = sorted(errs)
    mean = sum(errs)/len(errs)
    median = errs_sorted[len(errs_sorted)//2]
    print(f"Mean pixel error: {mean:.4f}")
    print(f"Median pixel error: {median:.4f}")
    print(f"Max pixel error: {max(errs):.4f}")
    # convert to arcseconds (your earlier value: 1 px ≈ 172.8 arcsec)
    px_to_arcsec = 172.8
    print(f"Mean angular error: {mean*px_to_arcsec:.1f} arcsec (~{mean*px_to_arcsec/3600:.4f}°)")
else:
    print("No matched rows!")
