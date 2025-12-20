import csv
import subprocess
from pathlib import Path
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = PROJECT_ROOT / "results"
RUN_DIR = RESULTS_ROOT / "exp_centroid_noise" / "run_000"

# ---------------------------------------------
# Project structure
# ---------------------------------------------
THIS_FILE = Path(__file__).resolve()

PROJECT_ROOT = THIS_FILE.parents[3]   # TheStarSensorProject
SCRIPTS_DIR  = PROJECT_ROOT / "src/scripts"
PIPELINE     = PROJECT_ROOT / "src/pipeline/startracker_pipeline.py"
SOLVER       = SCRIPTS_DIR / "run_attitude_solver.py"

RESULTS_DIR = PROJECT_ROOT / "results/exp_centroid_noise"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = RESULTS_DIR / "summary.csv"

# ---------------------------------------------
# Experiment settings
# ---------------------------------------------
NOISE_LEVELS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]

# ---------------------------------------------
# CSV header
# ---------------------------------------------
with open(SUMMARY_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "centroid_noise_px",
        "n_matches",
        "error_deg",
        "error_arcsec"
    ])

# ---------------------------------------------
# Main loop
# ---------------------------------------------
for noise in NOISE_LEVELS:
    print(f"\n=== centroid_noise = {noise} px ===")

    # Run pipeline
    subprocess.run([
        "python",
        str(PIPELINE),
        "--centroid-noise", str(noise)
    ], check=True)

    # Run solver and capture output
    output = subprocess.check_output(
    ["python", "-m", "src.scripts.run_attitude_solver"],
    cwd=PROJECT_ROOT,
    text=True
)


    err_deg = None
    err_arcsec = None

    for line in output.splitlines():
        if "deg" in line:
            err_deg = float(line.split("=")[1])
        if "arcsec" in line:
            err_arcsec = float(line.split("=")[1])

    # Count matches
    raw_match = RUN_DIR / "match_results.csv"
    refined_match = RUN_DIR / "match_results_refined.csv"

    if refined_match.exists():
        match_file = refined_match
    elif raw_match.exists():
        raise RuntimeError(
            "match_results_refined.csv missing — raw matches exist but are unreliable.\n"
            "Run refinement step before attitude estimation."
        )
    else:
        raise RuntimeError("No match results file found.")

    with open(match_file) as f:
        n_matches = sum(1 for _ in f) - 1

    # Write row
    with open(SUMMARY_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            noise,
            n_matches,
            err_deg,
            err_arcsec
        ])

    print(f"matches={n_matches}, error={err_arcsec:.2f} arcsec")

print("\n✓ Step 5.5B complete")
print("Saved:", SUMMARY_CSV)
