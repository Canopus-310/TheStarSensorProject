# src/attitude/estimate_attitude.py
import numpy as np
import csv

def estimate_R_from_matches(match_csv, projected_catalog_csv, fx, fy, cx, cy):
    """
    Estimate rotation R such that:
        v_cam ≈ R @ v_inertial
    """

    # ----------------------------
    # Load inertial vectors
    # ----------------------------
    inertial = []
    with open(projected_catalog_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            inertial.append([
                float(row["x"]),
                float(row["y"]),
                float(row["z"]),
            ])
    inertial = np.array(inertial)

    # ----------------------------
    # Load matched detections
    # ----------------------------
    v_cam = []
    v_in  = []

    with open(match_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            i = int(row["truth_idx"])

            u = float(row["det_x"])
            v = float(row["det_y"])

            xn = (u - cx) / fx
            yn = (v - cy) / fy

            cam = np.array([xn, yn, 1.0])
            cam /= np.linalg.norm(cam)

            v_cam.append(cam)
            v_in.append(inertial[i])

    v_cam = np.array(v_cam)
    v_in  = np.array(v_in)

    if len(v_cam) < 3:
        raise RuntimeError("Not enough matches for attitude estimation")

    # ----------------------------
    # Kabsch (ORTHONORMAL!)
    # ----------------------------
    H = v_cam.T @ v_in
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    return R
