"""
startracker_pipeline.py

Step 5 pipeline runner.
Coordinates projection → generation → detection → matching → refinement.
"""

import subprocess
from pathlib import Path
import sys


# --------------------------------------------------
# CONFIG
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = PROJECT_ROOT / "results" / "exp_centroid_noise" / "run_000"
RUN_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# STEP 1 — Project catalog
# --------------------------------------------------
def step1_project():
    print("\n=== STEP 1: PROJECT CATALOG ===")

    script = PROJECT_ROOT / "src/scripts/project_catalog_to_pixels_point_at_bright.py"

    subprocess.run([
        "python",
        str(script),
        "--out_dir", str(RUN_DIR)
    ], check=True)

    proj = RUN_DIR / "catalog_projected.csv"
    if not proj.exists():
        sys.exit("❌ catalog_projected.csv missing")

    print(f"✓ {proj}")


# --------------------------------------------------
# STEP 2 — Generate synthetic image
# --------------------------------------------------
def step2_generate(centroid_noise_px=0.0):
    print("\n=== STEP 2: GENERATE IMAGE ===")

    script = PROJECT_ROOT / "src/vision/catalog_based_generator.py"

    subprocess.run([
        "python",
        str(script),
        "--projected_catalog", str(RUN_DIR / "catalog_projected.csv"),
        "--out_dir", str(RUN_DIR),
        "--centroid-noise", str(centroid_noise_px)
    ], check=True)

    img = RUN_DIR / "catalog_sky_01_linear16.png"
    if not img.exists():
        sys.exit("❌ image not generated")

    print(f"✓ {img}")


# --------------------------------------------------
# STEP 3 — Detect stars
# --------------------------------------------------
def step3_detect():
    print("\n=== STEP 3: DETECT STARS ===")

    script = PROJECT_ROOT / "src/vision/detector.py"

    subprocess.run([
        "python",
        str(script),
        "--input", str(RUN_DIR / "catalog_sky_01_linear16.png"),
        "--out_dir", str(RUN_DIR)
    ], check=True)

    det = RUN_DIR / "detected_centroids.csv"
    if not det.exists():
        sys.exit("❌ detected_centroids.csv missing")

    print(f"✓ {det}")


# --------------------------------------------------
# STEP 4 — Match detections (raw)
# --------------------------------------------------
def step4_match():
    print("\n=== STEP 4: MATCH DETECTIONS ===")

    script = PROJECT_ROOT / "src/scripts/match_detections_to_truth.py"

    subprocess.run([
        "python",
        str(script),
        "--det", str(RUN_DIR / "detected_centroids.csv"),
        "--truth", str(RUN_DIR / "catalog_sky_01_truth.csv"),
        "--out", str(RUN_DIR / "match_results.csv"),
        "--thresh", "8"
    ], check=True)

    out = RUN_DIR / "match_results.csv"
    if not out.exists():
        sys.exit("❌ match_results.csv missing")

    print(f"✓ {out}")


# --------------------------------------------------
# STEP 4.5 — Refine matches
# --------------------------------------------------
def step4_refine():
    print("\n=== STEP 4.5: REFINE MATCHES ===")

    script = PROJECT_ROOT / "src/scripts/match_refine.py"

    subprocess.run([
        "python",
        str(script),
        "--input", str(RUN_DIR / "match_results.csv"),
        "--out", str(RUN_DIR / "match_results_refined.csv"),
        "--max_err_px", "5.0"
    ], check=True)

    refined = RUN_DIR / "match_results_refined.csv"
    if not refined.exists():
        sys.exit("❌ match_results_refined.csv missing")

    print(f"✓ {refined}")


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------
def main():
    print("\n======================================")
    print("        STAR TRACKER PIPELINE")
    print("======================================")

    step1_project()
    step2_generate(centroid_noise_px=0.0)
    step3_detect()
    step4_match()
    step4_refine()

    print("\n✓ Pipeline completed successfully")
    print(f"Results in: {RUN_DIR}\n")


if __name__ == "__main__":
    main()
