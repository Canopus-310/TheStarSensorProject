# =========================
# Imports
# =========================
import numpy as np
from scipy.spatial import KDTree
from numpy.linalg import svd

# =========================
# Camera parameters
# =========================
height = 480
width = 640
fov_x = 30
fov_y = 30

fx = (width/2) / np.tan(np.deg2rad(fov_x/2))
fy = (height/2) / np.tan(np.deg2rad(fov_y/2))
cx = width / 2
cy = height / 2

# =========================
# Pixel → camera unit vector
# =========================
def pixel_to_camera(u, v):
    x = (u - cx) / fx
    y = (v - cy) / fy
    z = 1.0
    vec = np.array([x, y, z])
    return vec / np.linalg.norm(vec)

# =========================
# Angular distance (safe)
# =========================
def angular_distance(a, b):
    return np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))

# =========================
# Triangle signature
# =========================
def vectors_to_angles(a, b, c):
    angles = [
        angular_distance(a, b),
        angular_distance(b, c),
        angular_distance(c, a)
    ]
    return np.sort(angles)

# =========================
# Tolerances (radians)
# =========================
centroid_angle_error = np.deg2rad(0.14)  # ~518 arcsec
tolerance = max(2 * centroid_angle_error, np.deg2rad(0.01))
residual_tolerance = 3 * centroid_angle_error

# =========================
# STEP 5.1 — BUILD CATALOG TRIANGLES
# =========================
def build_catalog_triangles(catalog_vectors, star_ids, max_stars=200):
    catalog_vectors = catalog_vectors[:max_stars]
    star_ids = star_ids[:max_stars]

    signatures = []
    metadata = []

    n = len(catalog_vectors)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a, b, c = catalog_vectors[i], catalog_vectors[j], catalog_vectors[k]
                angles = vectors_to_angles(a, b, c)

                if angles[0] < 5 * centroid_angle_error:
                    continue  # degenerate

                signatures.append(angles)
                metadata.append((star_ids[i], star_ids[j], star_ids[k]))

    return np.array(signatures), metadata

# =========================
# STEP 5.2 — BUILD KD-TREE
# =========================
def build_kdtree(triangle_signatures):
    return KDTree(triangle_signatures)

# =========================
# STEP 5.3 — MATCH IMAGE TRIANGLE
# =========================
def match_triangle(image_triangle_angles, kdtree, catalog_sigs, catalog_meta, k=30):
    dists, idxs = kdtree.query(image_triangle_angles, k=k)

    matches = []
    for idx in np.atleast_1d(idxs):
        if np.all(np.abs(catalog_sigs[idx] - image_triangle_angles) < tolerance):
            matches.append(catalog_meta[idx])

    return matches

# =========================
# STEP 5.4 — WAHBA (SVD / KABSCH)
# =========================
def solve_wahba_svd(camera_vecs, inertial_vecs):
    B = np.zeros((3,3))
    for d, c in zip(camera_vecs, inertial_vecs):
        B += np.outer(d, c)

    U, _, Vt = svd(B)
    R = U @ np.diag([1,1,np.linalg.det(U @ Vt)]) @ Vt
    return R

# =========================
# STEP 5.5 — RESIDUALS
# =========================
def compute_residuals(R, camera_vecs, inertial_vecs):
    residuals = []
    for d, c in zip(camera_vecs, inertial_vecs):
        residuals.append(angular_distance(d, R @ c))
    return np.array(residuals)

# =========================
# STEP 5.6 — RANSAC ATTITUDE
# =========================
def estimate_attitude_ransac(camera_vecs, inertial_vecs, iterations=50):
    best_R = None
    best_inliers = []

    n = len(camera_vecs)
    if n < 3:
        return None, []

    for _ in range(iterations):
        idx = np.random.choice(n, 3, replace=False)
        R = solve_wahba_svd(camera_vecs[idx], inertial_vecs[idx])

        residuals = compute_residuals(R, camera_vecs, inertial_vecs)
        inliers = residuals < residual_tolerance

        if np.sum(inliers) > np.sum(best_inliers):
            best_R = R
            best_inliers = inliers

    if best_R is None:
        return None, []

    # refine with inliers
    R_refined = solve_wahba_svd(camera_vecs[best_inliers], inertial_vecs[best_inliers])
    return R_refined, best_inliers

# =========================
# STEP 5.7 — ATTITUDE ERROR
# =========================
def rotation_error(R_true, R_est):
    delta = R_true.T @ R_est
    angle = np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1))
    return np.rad2deg(angle) * 3600  # arcsec
