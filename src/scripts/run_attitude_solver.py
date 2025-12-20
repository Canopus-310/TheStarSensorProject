# src/scripts/run_attitude_solver.py
import numpy as np
from pathlib import Path
from src.attitude.estimate_attitude import estimate_R_from_matches

def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    RUN_DIR = PROJECT_ROOT / "results/exp_centroid_noise/run_000"

    MATCH_CSV = RUN_DIR / "match_results_refined.csv"
    PROJ_CSV  = RUN_DIR / "catalog_projected.csv"
    R_TRUE    = np.loadtxt(RUN_DIR / "R_true.txt")

    WIDTH = 640
    HEIGHT = 480
    FOV_X_DEG = 30.0

    fx = (WIDTH / 2) / np.tan(np.deg2rad(FOV_X_DEG / 2))
    fy = fx
    cx = WIDTH / 2
    cy = HEIGHT / 2

    R_est = estimate_R_from_matches(
        match_csv=str(MATCH_CSV),
        projected_catalog_csv=str(PROJ_CSV),
        fx=fx, fy=fy, cx=cx, cy=cy
    )

    R_err = R_est @ R_TRUE.T
    trace = np.clip(np.trace(R_err), -1.0, 3.0)
    theta = np.arccos((trace - 1) / 2)

    print("\n=== ATTITUDE SOLUTION ===")
    print("Estimated R:\n", R_est)
    print("det(R) =", np.linalg.det(R_est))
    print(f"Attitude error: {np.degrees(theta):.6f} deg = {np.degrees(theta)*3600:.2f} arcsec\n")

if __name__ == "__main__":
    main()
