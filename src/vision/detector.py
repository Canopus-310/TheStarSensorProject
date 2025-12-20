"""
detector.py

Pure star detection + centroiding.
NO pipeline logic.
NO subprocess calls.
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
import pandas as pd


def detect_stars(
    image_path: Path,
    out_dir: Path,
    threshold: float = 5.0,
    min_area: int = 5,
):
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    img = img.astype(np.float32)

    # Simple thresholding (you already improved this earlier)
    mask = img > threshold

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8
    )

    rows = []
    for i in range(1, num_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        cx, cy = centroids[i]
        rows.append({
            "x": cx,
            "y": cy,
            "area": area
        })

    df = pd.DataFrame(rows)
    out_path = out_dir / "detected_centroids.csv"
    df.to_csv(out_path, index=False)

    print(f"[detector] Detected {len(df)} stars")
    print(f"[detector] Wrote: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Input image")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--min_area", type=int, default=5)

    args = parser.parse_args()

    detect_stars(
        image_path=Path(args.input),
        out_dir=Path(args.out_dir),
        threshold=args.threshold,
        min_area=args.min_area
    )


if __name__ == "__main__":
    main()
