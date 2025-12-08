import argparse, numpy as np, cv2, csv, os

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True)
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--min_area", type=int, default=6)
    ap.add_argument("--snr_thresh", type=float, default=1.5)
    ap.add_argument("--centroid_win", type=int, default=7)
    return ap.parse_args()


def centroid(img, pts, win):
    """
    Subpixel centroid using weighted mean inside a local window.
    pts includes (x, y, area, peak)
    """
    h, w = img.shape
    out = []
    r = win // 2

    for (x, y, area, peak) in pts:
        xi = int(round(x))
        yi = int(round(y))

        if xi < r or yi < r or xi + r >= w or yi + r >= h:
            continue

        patch = img[yi-r:yi+r+1, xi-r:xi+r+1].astype(float)

        # Weighted centroid
        Y, X = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
        m00 = patch.sum()
        if m00 <= 1e-9:
            continue

        cx = (patch * X).sum() / m00
        cy = (patch * Y).sum() / m00

        out.append((xi + (cx - r), yi + (cy - r), area, peak))

    return out


def main():
    args = parse_args()

    # Load image
    img = cv2.imread(args.input, cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise RuntimeError(f"Failed to load image: {args.input}")

    img = img.astype(np.float32)
    h, w = img.shape

    # -------------------------
    # FIX 3: Adaptive threshold
    # -------------------------
    if args.threshold is None:
        p20 = np.percentile(img, 20)           # low percentile background
        sigma_est = max(1.0, p20 / 2.0)        # rough noise estimate
        T = p20 + 4.0 * sigma_est              # safe threshold
        print(f"[detector] Auto-threshold: {T:.2f} (p20={p20:.2f}, sigma≈{sigma_est:.2f})")
    else:
        T = args.threshold
        print(f"[detector] Manual threshold: {T:.2f}")

    # Threshold → mask
    _, mask = cv2.threshold(img, T, 1, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)

    # Connected components
    nlab, lab = cv2.connectedComponents(mask)

    raw_pts = []
    for lab_id in range(1, nlab):
        ys, xs = np.where(lab == lab_id)
        area = len(xs)
        if area < args.min_area:
            continue
        peak = img[ys, xs].max()
        raw_pts.append((xs.mean(), ys.mean(), area, peak))

    # -------------------------
    # FIX 4: Correct SNR filter
    # -------------------------
    filtered = []
    win = args.centroid_win
    r = win // 2

    # estimate *global* noise for SNR floor
    p10 = np.percentile(img, 10)
    noise_sigma = max(1.0, p10 / 2.0)

    for (x, y, area, peak) in raw_pts:
        xi = int(round(x))
        yi = int(round(y))

        if xi < r or yi < r or xi + r >= w or yi + r >= h:
            continue

        patch = img[yi-r:yi+r+1, xi-r:xi+r+1]
        local_bg = np.median(patch)

        snr = (peak - local_bg) / (noise_sigma + 1e-6)

        if snr >= args.snr_thresh:
            filtered.append((x, y, area, peak))

    # Subpixel centroiding
    det2 = centroid(img, filtered, args.centroid_win)

    # Save results
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "detected_centroids.csv")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "area", "peak"])
        for (x, y, a, p) in det2:
            w.writerow([f"{x:.3f}", f"{y:.3f}", a, p])

    print("Detections:", len(det2))
    print("Wrote:", out_csv)


if __name__ == "__main__":
    main()
