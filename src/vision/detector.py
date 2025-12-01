import os
import cv2
import numpy as np

# Resolve project root relative to this file
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def threshold_stars(
    input_path: str = None,
    output_thresh_path: str = None,
    use_otsu: bool = False,
    manual_T: int = 65,
) -> np.ndarray:
    """
    Load the synthetic sky image and apply a global threshold
    to produce a binary mask of candidate stars.
    """
    if input_path is None:
        input_path = os.path.join(RESULTS_DIR, "synthetic_sky_01.png")
    if output_thresh_path is None:
        output_thresh_path = os.path.join(
            RESULTS_DIR, f"synthetic_sky_01_thresh_T{manual_T}.png"
        )

    print("Using input path:", input_path)
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {input_path}")

    print(f"Image stats: min={img.min()}, max={img.max()}, mean={img.mean():.2f}")

    # Debug: fraction of pixels above various thresholds
    for T in [10, 20, 27, 40, 55, 60, 80, 100]:
        frac = np.mean(img > T)
        print(f"T={T:3d} -> {frac*100:6.2f}% pixels above threshold")

    if use_otsu:
        retval, thresh_img = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        print("Otsu chose threshold:", retval)
    else:
        retval, thresh_img = cv2.threshold(
            img, manual_T, 255, cv2.THRESH_BINARY
        )
        print("Manual threshold used:", manual_T)

    ensure_dir(os.path.dirname(output_thresh_path))
    cv2.imwrite(output_thresh_path, thresh_img)
    print("Saved binary threshold image to:", output_thresh_path)

    return thresh_img


def clean_binary_mask(
    binary_img: np.ndarray,
    min_area: int = 5,
    output_path: str = None,
) -> np.ndarray:
    """
    Remove small white blobs (noise specks) from a binary image
    using connected components + area filtering.
    """
    if output_path is None:
        output_path = os.path.join(RESULTS_DIR, "synthetic_sky_01_clean.png")

    # Ensure binary is 0/255 uint8
    bin_img = (binary_img > 0).astype(np.uint8) * 255

    # connectivity=8 → diagonal pixels also considered connected
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bin_img, connectivity=8
    )

    # stats: [label, 5 stats] -> [x, y, width, height, area]
    areas = stats[:, cv2.CC_STAT_AREA]

    # Create an empty mask
    clean = np.zeros_like(bin_img)

    kept = 0
    for label in range(1, num_labels):  # skip background (0)
        if areas[label] >= min_area:
            clean[labels == label] = 255
            kept += 1

    ensure_dir(os.path.dirname(output_path))
    cv2.imwrite(output_path, clean)
    print(f"Cleaned mask saved to: {output_path}")
    print(f"Total labels: {num_labels}, kept (area >= {min_area}): {kept}")

    return clean


def detect_centroids(
    clean_binary_img: np.ndarray,
    original_img_path: str = None,
    overlay_output_path: str = None,
    csv_output_path: str = None,
) -> list[tuple[float, float]]:
    """
    From a cleaned binary image, find connected components and
    return their centroids. Also optionally draw them on the
    original image and save a CSV.
    """
    if original_img_path is None:
        original_img_path = os.path.join(RESULTS_DIR, "synthetic_sky_01.png")
    if overlay_output_path is None:
        overlay_output_path = os.path.join(RESULTS_DIR, "synthetic_sky_01_detected.png")
    if csv_output_path is None:
        csv_output_path = os.path.join(RESULTS_DIR, "synthetic_sky_01_centroids.csv")

    # Ensure 0/255 uint8 binary
    bin_img = (clean_binary_img > 0).astype(np.uint8) * 255

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bin_img, connectivity=8
    )

    # Label 0 = background, skip it
    detected_centroids = []
    for label in range(1, num_labels):
        cx, cy = centroids[label]  # note: (x, y) = (column, row)
        detected_centroids.append((float(cx), float(cy)))

    print(f"Detected {len(detected_centroids)} centroids")

    # Draw on original image for visualization
    orig = cv2.imread(original_img_path, cv2.IMREAD_GRAYSCALE)
    if orig is None:
        raise FileNotFoundError(f"Could not load original image at {original_img_path}")

    orig_color = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
    for (cx, cy) in detected_centroids:
        cv2.circle(orig_color, (int(round(cx)), int(round(cy))), 3, (0, 0, 255), 1)

    ensure_dir(os.path.dirname(overlay_output_path))
    cv2.imwrite(overlay_output_path, orig_color)
    print(f"Saved centroid overlay image to: {overlay_output_path}")

    # Save CSV
    ensure_dir(os.path.dirname(csv_output_path))
    with open(csv_output_path, "w") as f:
        f.write("id,x,y\n")
        for idx, (cx, cy) in enumerate(detected_centroids):
            f.write(f"{idx},{cx:.3f},{cy:.3f}\n")

    print(f"Saved centroids CSV to: {csv_output_path}")

    return detected_centroids


if __name__ == "__main__":
    # 1. Threshold
    thresh = threshold_stars(use_otsu=False, manual_T=55)

    # 2. Clean noise specks (you liked min_area=3, but 5 is also fine)
    clean = clean_binary_mask(thresh, min_area=3)

    # 3. Detect centroids and visualize
    centroids = detect_centroids(clean)
    print("Example centroid:", centroids[0] if centroids else "None")
