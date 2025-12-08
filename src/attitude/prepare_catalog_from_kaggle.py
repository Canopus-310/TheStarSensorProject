#!/usr/bin/env python3
# src/attitude/prepare_catalog_from_kaggle.py
"""
Robust converter: Kaggle Hipparcos-like CSV -> data/catalog_unit_vectors.csv
Usage:
  python src/attitude/prepare_catalog_from_kaggle.py --in data/hipparcos_subset.csv --out data/catalog_unit_vectors.csv
"""
import csv, math, argparse, os, sys, re

def parse_hms_to_float(s):
    s = str(s).strip()
    if s == "": return None
    s = s.replace('h',' ').replace('m',' ').replace('s',' ').replace('°',' ').replace("'", " ").strip()
    # split by non-digit separators
    parts = re.split(r'[:\s]+', s)
    parts = [p for p in parts if p!='']
    if len(parts) == 0:
        try: return float(s)
        except: return None
    try:
        if len(parts) == 1:
            return float(parts[0])
        h = float(parts[0])
        m = float(parts[1]) if len(parts) > 1 else 0.0
        sec = float(parts[2]) if len(parts) > 2 else 0.0
        return abs(h) + m/60.0 + sec/3600.0
    except:
        try:
            return float(s)
        except:
            return None

def parse_ra_field(v):
    if v is None: return None
    s = str(v).strip()
    if s == "": return None
    # common RA formats: "hh:mm:ss.s", "hh mm ss", decimal hours, decimal degrees
    # If string contains ':' or 'h' or ' ' treat as HMS
    if any(ch in s for ch in [':','h','m','s']):
        val_hours = parse_hms_to_float(s)
        if val_hours is None: return None
        # if seems like hours (<=24) convert to degrees
        if 0.0 <= val_hours <= 24.0: return val_hours * 15.0
        return val_hours
    # otherwise numeric
    try:
        val = float(s)
    except:
        return None
    # Heuristic: if 0..24 likely hours
    if 0.0 <= val <= 24.0:
        return val * 15.0
    return val

def parse_dec_field(v):
    if v is None: return None
    s = str(v).strip()
    if s == "": return None
    sign = 1.0
    if s.startswith('-'):
        sign = -1.0
        s = s[1:].strip()
    if s.startswith('+'):
        s = s[1:].strip()
    if any(ch in s for ch in [':','d','°',"'",' ']):
        deg = parse_hms_to_float(s)
        if deg is None:
            try: return float(s)*sign
            except: return None
        return sign * deg
    try:
        return float(s)
    except:
        return None

def guess_columns(header):
    # normalize header list for checks
    hdr_lower = [h.lower() for h in header]
    ra_col = None; dec_col = None; mag_col = None

    # 1) Prefer explicit decimal-degree columns if present
    for i,h in enumerate(hdr_lower):
        if ra_col is None and h in ('radeg','ra_deg','ra_deg_j2000','ra_deg.'):
            ra_col = header[i]
        if dec_col is None and h in ('dedeg','dec_deg','dec_deg_j2000','dec_deg.','de_deg'):
            dec_col = header[i]
        if mag_col is None and h in ('vmag','v_mag','mag','phot_g_mean_mag'):
            mag_col = header[i]

    # 2) If decimals not found, fall back to common names (including RAhms / DEdms)
    if ra_col is None or dec_col is None:
        ra_candidates = ['rahms', 'ra_hms', 'ra_h', 'ra_hours', 'rahrs', 'radeg', 'ra', 'raj2000', 'ra_j2000']
        dec_candidates = ['dedms', 'decdms', 'dec_dms', 'dec_d', 'decdeg', 'dec_deg', 'dec', 'de', 'decj2000', 'dec_j2000']
        mag_candidates = ['vmag', 'mag', 'v', 'phot_g_mean_mag', 'hp_mag', 'hp']
        for i,h in enumerate(hdr_lower):
            if ra_col is None:
                for cand in ra_candidates:
                    if cand in h or h in cand:
                        ra_col = header[i]; break
            if dec_col is None:
                for cand in dec_candidates:
                    if cand in h or h in cand:
                        dec_col = header[i]; break
            if mag_col is None:
                for cand in mag_candidates:
                    if cand in h or h in cand:
                        mag_col = header[i]; break
            if ra_col and dec_col and mag_col:
                break

    # 3) Final fallback: exact matches
    if ra_col is None:
        for k in header:
            if k.lower() in ('rahms','ra','ra_deg','radeg','raj2000'):
                ra_col = k; break
    if dec_col is None:
        for k in header:
            if k.lower() in ('dedms','ded','dec','dec_deg','decdeg','decdms','de'):
                dec_col = k; break
    if mag_col is None:
        for k in header:
            if k.lower() in ('vmag','mag','v','hp','phot_g_mean_mag'):
                mag_col = k; break

    return ra_col, dec_col, mag_col



def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='infile', required=True)
    p.add_argument('--out', dest='outfile', default='data/catalog_unit_vectors.csv')
    args = p.parse_args()
    IN = args.infile; OUT = args.outfile
    if not os.path.exists(IN):
        print("Input missing:", IN); sys.exit(1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    with open(IN, newline='', encoding='utf-8', errors='replace') as f:
        r = csv.DictReader(f)
        hdr = r.fieldnames
        print("Header detected:", hdr)
        ra_col, dec_col, mag_col = guess_columns(hdr)
        print("Guessed columns -> RA:", ra_col, " DEC:", dec_col, " MAG:", mag_col)
        out_rows = []
        bad = 0; total = 0
        for row in r:
            total += 1
            ra_raw = row.get(ra_col) if ra_col else (row.get('RAhms') or row.get('RAh'))
            dec_raw = row.get(dec_col) if dec_col else (row.get('DECdms') or row.get('DEd') or row.get('DE'))
            mag_raw = row.get(mag_col) if mag_col else row.get('Vmag') or row.get('mag') or row.get('VMag')
            ra_deg = parse_ra_field(ra_raw)
            dec_deg = parse_dec_field(dec_raw)
            try:
                mag = float(mag_raw) if (mag_raw is not None and str(mag_raw).strip()!='') else 9.0
            except:
                mag = 9.0
            if ra_deg is None or dec_deg is None:
                bad += 1
                continue
            ra = math.radians(ra_deg); dec = math.radians(dec_deg)
            x = math.cos(dec) * math.cos(ra)
            y = math.cos(dec) * math.sin(ra)
            z = math.sin(dec)
            out_rows.append([str(total), f"{ra_deg:.8f}", f"{dec_deg:.8f}", f"{mag:.3f}",
                             f"{x:.12g}", f"{y:.12g}", f"{z:.12g}"])
    with open(OUT, "w", newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["id","ra_deg","dec_deg","mag","x","y","z"])
        w.writerows(out_rows)
    print(f"Wrote {len(out_rows)} rows to {OUT}  (skipped {bad} malformed rows out of {total})")

if __name__ == "__main__":
    main()
