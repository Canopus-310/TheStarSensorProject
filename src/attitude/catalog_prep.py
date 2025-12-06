# src/attitude/catalog_prep.py
"""
Convert a (Hipparcos-like) input CSV with RA/Dec (+ optional proper motions)
into a CSV of unit vectors (x,y,z) suitable for the rest of the pipeline.

Usage:
    python src/attitude/catalog_prep.py --input data/hipparcos_subset.csv --output data/catalog_unit_vectors.csv --epoch 1991.25

The script is robust to a few input formats:
 - RA can be provided as "RA" string "hh:mm:ss.s" or separated ra_h, ra_m, ra_s or numeric hours/degrees
 - Dec can be provided as "DEC" string "+dd:mm:ss.s" or separated dec_deg, dec_min, dec_sec or numeric degrees
 - Proper motions fields can be named pm_ra_cosdec / pm_dec or pmRA / pmDE etc (values in mas/yr)
 - Magnitude field is optional
"""
import csv
import math
import argparse
from datetime import datetime, timezone

# ---------- helpers ----------
def hms_to_deg(h, m, s):
    return (float(h) + float(m) / 60.0 + float(s) / 3600.0) * 15.0

def dms_to_deg(d, m, s):
    # d may carry sign
    d = float(d)
    sign = -1.0 if d < 0 else 1.0
    return sign * (abs(d) + float(m) / 60.0 + float(s) / 3600.0)

def parse_hms_string(s):
    s = s.strip()
    if not s:
        return None
    # common separators: ":", " ", "h m s"
    for ch in ['h', 'm', 's']:
        s = s.replace(ch, ' ')
    s = s.replace(':', ' ').strip()
    parts = s.split()
    if len(parts) == 1:
        # maybe numeric hours or degrees
        try:
            val = float(parts[0])
            # heuristic: if <= 24 -> hours else -> degrees
            return val * 15.0 if val <= 24 else val
        except:
            return None
    while len(parts) < 3:
        parts.append('0')
    h, m, sec = parts[:3]
    return hms_to_deg(h, m, sec)

def parse_dms_string(s):
    s = s.strip()
    if not s:
        return None
    # allow leading +/-
    sign = 1
    if s[0] in ['+', '-']:
        if s[0] == '-':
            sign = -1
        s = s[1:].strip()
    for ch in ['d', '°', '\'', '"', 'm', 's']:
        s = s.replace(ch, ' ')
    s = s.replace(':', ' ').strip()
    parts = s.split()
    if len(parts) == 1:
        try:
            return float(parts[0]) * sign
        except:
            return None
    while len(parts) < 3:
        parts.append('0')
    deg, m, sec = parts[:3]
    try:
        return dms_to_deg(sign * float(deg), float(m), float(sec))
    except:
        return None

# ---------- RA/Dec -> cartesian ----------
def radec_to_unit_vector(ra_deg, dec_deg):
    ra = math.radians(float(ra_deg))
    dec = math.radians(float(dec_deg))
    x = math.cos(dec) * math.cos(ra)
    y = math.cos(dec) * math.sin(ra)
    z = math.sin(dec)
    return x, y, z

# ---------- proper motion propagation ----------
# pm_ra_cosdec and pm_dec in milliarcseconds/year (mas/yr)
def apply_proper_motion(ra_deg, dec_deg, pm_ra_cosdec_masyr, pm_dec_masyr, epoch_from, epoch_to):
    dt = float(epoch_to) - float(epoch_from)
    # convert mas -> degrees: 1 deg = 3.6e6 mas
    dra_deg = (float(pm_ra_cosdec_masyr) / 3.6e6) * dt if pm_ra_cosdec_masyr is not None else 0.0
    ddec_deg = (float(pm_dec_masyr) / 3.6e6) * dt if pm_dec_masyr is not None else 0.0
    ra_new = float(ra_deg) + dra_deg
    dec_new = float(dec_deg) + ddec_deg
    # normalize RA into [0,360)
    ra_new = ra_new % 360.0
    # clamp dec to valid range
    if dec_new > 90.0:
        dec_new = 90.0
    if dec_new < -90.0:
        dec_new = -90.0
    return ra_new, dec_new

# ---------- main converter ----------
def convert_catalog(input_csv, output_csv, catalog_epoch=1991.25, target_epoch=None, id_field_hint=None):
    # compute default target epoch = current decimal year (UTC)
    if target_epoch is None:
        now = datetime.now(timezone.utc)
        year = now.year
        start_of_year = datetime(year, 1, 1, tzinfo=timezone.utc)
        days = (now - start_of_year).total_seconds() / 86400.0
        target_epoch = year + days / 365.25

    with open(input_csv, newline='', encoding='utf-8', errors='replace') as f_in, \
         open(output_csv, 'w', newline='', encoding='utf-8') as f_out:

        reader = csv.DictReader(f_in)
        fieldnames = ['id', 'ra_deg', 'dec_deg', 'mag', 'x', 'y', 'z']
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            # --- id ---
            uid = ''
            if id_field_hint and id_field_hint in row:
                uid = row[id_field_hint]
            else:
                for k in ('id', 'ID', 'HIP', 'hip', 'source_id'):
                    if k in row and row[k] not in (None, ''):
                        uid = row[k]
                        break

            # ---------- parse RA ----------
            ra_deg = None
            # 1) separate fields ra_h, ra_m, ra_s
            if row.get('ra_h') and row.get('ra_m') and row.get('ra_s'):
                try:
                    ra_deg = hms_to_deg(row['ra_h'], row['ra_m'], row['ra_s'])
                except:
                    ra_deg = None
            # 2) common combined fields RA / ra
            if ra_deg is None:
                ra_field = (row.get('ra') or row.get('RA') or row.get('Ra') or '').strip()
                if ra_field:
                    ra_deg = parse_hms_string(ra_field)
            # 3) sometimes RA is in columns RAh, RAm, RAs (Hipparcos-style uppercase)
            if ra_deg is None and row.get('RAh') and row.get('RAm') and row.get('RAs'):
                try:
                    ra_deg = hms_to_deg(row['RAh'], row['RAm'], row['RAs'])
                except:
                    ra_deg = None
            # 4) sometimes RA is numeric degrees or hours in a numeric column named 'RA_deg' or 'raj2000'
            if ra_deg is None:
                for cand in ('ra_deg', 'RA_deg', 'raj2000', 'RAJ2000', 'ra_deg2000', 'ra_deg_fk5'):
                    if cand in row and row[cand] not in (None, ''):
                        try:
                            val = float(row[cand])
                            # heuristic: if <=24 treat as hours
                            ra_deg = val * 15.0 if val <= 24.0 else val
                            break
                        except:
                            pass
            # fallback 0
            if ra_deg is None:
                ra_deg = 0.0

            # ---------- parse Dec ----------
            dec_deg = None
            # 1) separate fields dec_deg, dec_min, dec_sec
            if row.get('dec_deg') and row.get('dec_min') and row.get('dec_sec'):
                try:
                    dec_deg = dms_to_deg(row['dec_deg'], row['dec_min'], row['dec_sec'])
                except:
                    dec_deg = None
            # 2) common combined fields DEC / dec
            if dec_deg is None:
                dec_field = (row.get('dec') or row.get('DEC') or row.get('Dec') or '').strip()
                if dec_field:
                    dec_deg = parse_dms_string(dec_field)
            # 3) Hipparcos-style DEd, DEm, DEs
            if dec_deg is None and row.get('DEd') and row.get('DEm') and row.get('DEs'):
                try:
                    dec_deg = dms_to_deg(row['DEd'], row['DEm'], row['DEs'])
                except:
                    dec_deg = None
            # 4) numeric fields
            if dec_deg is None:
                for cand in ('dec_deg', 'DEC_deg', 'dec2000', 'DEJ2000'):
                    if cand in row and row[cand] not in (None, ''):
                        try:
                            dec_deg = float(row[cand])
                            break
                        except:
                            pass
            # fallback
            if dec_deg is None:
                dec_deg = 0.0

            # ---------- magnitude ----------
            mag = ''
            for cand in ('Vmag', 'V', 'mag', 'VmagJ', 'Vmag_mean'):
                if cand in row and row[cand] not in (None, ''):
                    mag = row[cand]
                    break

            # ---------- epoch for this row ----------
            row_epoch = catalog_epoch
            if 'epoch' in row and row['epoch'] not in (None, ''):
                try:
                    row_epoch = float(row['epoch'])
                except:
                    row_epoch = catalog_epoch

            # ---------- proper motion ----------
            # try multiple possible names (mas/yr)
            pm_ra_val = None
            pm_dec_val = None
            # RA proper motion often stored as pmRA or pm_ra_cosdec or pmRAcosd
            for cand in ('pm_ra_cosdec', 'pmRA', 'pmRA_cosdec', 'pmRAcosd', 'pmra', 'pmra_cosdec'):
                if cand in row and row[cand] not in (None, ''):
                    try:
                        pm_ra_val = float(row[cand])
                        break
                    except:
                        pm_ra_val = None
            for cand in ('pm_dec', 'pmDE', 'pmDE_cosdec', 'pmdec', 'pmde'):
                if cand in row and row[cand] not in (None, ''):
                    try:
                        pm_dec_val = float(row[cand])
                        break
                    except:
                        pm_dec_val = None
            # Some catalogs have separate pmRA (without cosdec) and a separate flag; commonly Hipparcos provides pmRA* (i.e. pm_ra_cosdec)
            if pm_ra_val is None:
                # try pmRA (might be in mas/yr)
                if 'pmRA' in row and row['pmRA'] not in (None, ''):
                    try:
                        pm_ra_val = float(row['pmRA'])
                    except:
                        pm_ra_val = None

            if pm_dec_val is None:
                if 'pmDE' in row and row['pmDE'] not in (None, ''):
                    try:
                        pm_dec_val = float(row['pmDE'])
                    except:
                        pm_dec_val = None

            # ---------- apply proper motion propagation ----------
            try:
                ra_use, dec_use = apply_proper_motion(ra_deg, dec_deg, pm_ra_val or 0.0, pm_dec_val or 0.0, row_epoch, target_epoch)
            except Exception:
                ra_use, dec_use = ra_deg, dec_deg

            # ---------- convert to unit vector ----------
            x, y, z = radec_to_unit_vector(ra_use, dec_use)

            writer.writerow({
                'id': uid,
                'ra_deg': f"{ra_use:.8f}",
                'dec_deg': f"{dec_use:.8f}",
                'mag': mag,
                'x': f"{x:.12g}",
                'y': f"{y:.12g}",
                'z': f"{z:.12g}"
            })

    print("Wrote catalog unit vectors to:", output_csv)
    print(f"Catalog epoch was {catalog_epoch}, propagated to target epoch {target_epoch:.5f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True, help="Input CSV with RA/Dec")
    ap.add_argument("--output", "-o", required=True, help="Output catalog unit vectors CSV")
    ap.add_argument("--epoch", type=float, default=1991.25, help="Catalog epoch (default Hipparcos J1991.25)")
    ap.add_argument("--target", type=float, default=None, help="Target epoch to propagate to (decimal year). Default = now.")
    ap.add_argument("--id-field", default=None, help="Column name to use as id if not standard")
    args = ap.parse_args()
    convert_catalog(args.input, args.output, catalog_epoch=args.epoch, target_epoch=args.target, id_field_hint=args.id_field)
