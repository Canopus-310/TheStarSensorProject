# src/vision/starfield_generator.py

import os
import numpy as np
import cv2

def generate_starfield(
    width=640,
    height=480,
    n_stars=2000,
    psf_sigma=1.6,        # controls blur size in pixels
    max_flux= 1.0,         # brightest star intensity (before blur)
    read_noise_std=0.01,  # Gaussian noise std (relative)
    output_path="results/synthetic_sky_01.png"
):
    # 1. Make sure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 2. Empty black image (float32 for precision)
    img = np.zeros((height, width), dtype=np.float32)

    # 3. Random star positions (continuous coordinates)
    # x in [0, width), y in [0, height)
    x_true = np.random.uniform(0, width, size=n_stars)
    y_true = np.random.uniform(0, height, size=n_stars)

    # 4. Random brightnesses (fluxes)
    # Use something like a power-law so most stars are faint, few are bright
    # u in (0,1] then flux ~ u^(-alpha) truncated
    alpha = 1.5
    u = np.random.uniform(0.01, 1.0, size=n_stars)
    fluxes = max_flux * (u ** (-alpha))
    # Normalize so the brightest is max_flux
    fluxes /= fluxes.max()
    fluxes *= max_flux

    # 5. Deposit each star's flux into nearest pixel
    # (simple approximation; PSF will smooth it)
    x_pix = np.clip(np.round(x_true).astype(int), 0, width - 1)
    y_pix = np.clip(np.round(y_true).astype(int), 0, height - 1)

    for i in range(n_stars):
     img[y_pix[i], x_pix[i]] += fluxes[i]

    # 6. Apply Gaussian PSF (optics blur)
    # kernel size chosen ~ 6*sigma (odd number)
    ksize = int(6 * psf_sigma) | 1  # make sure it's odd
    img_blurred = cv2.GaussianBlur(img, (ksize, ksize), psf_sigma)

    # 7. Add sensor noise
    # 7a. Poisson-like noise: simulate photon counting
    # Scale to some "photon count" range first
    photon_scale = 5000.0
    img_photons = img_blurred * photon_scale
    img_photons[img_photons < 0] = 0
    noisy_photons = np.random.poisson(img_photons).astype(np.float32)
    img_noisy = noisy_photons / photon_scale

    # 7b. Add Gaussian read noise
    gauss_noise = np.random.normal(
        loc=0.0,
        scale=read_noise_std,
        size=img_noisy.shape
    ).astype(np.float32)
    img_noisy += gauss_noise

    # 8. Clip + normalize to 0–255 (8-bit image)
    img_noisy[img_noisy < 0] = 0
    # prevent division by zero
    max_val = img_noisy.max() if img_noisy.max() > 0 else 1.0
    img_norm = img_noisy / max_val
    img_uint8 = (img_norm * 255.0).clip(0, 255).astype(np.uint8)

    # 9. Save the image
    cv2.imwrite(output_path, img_uint8)

    # Optional: return useful info for debugging / later use
    return {
        "image": img_uint8,
        "x_true": x_true,
        "y_true": y_true,
        "fluxes": fluxes
    }


if __name__ == "__main__":
    info = generate_starfield()
    print("Saved:", "results/synthetic_sky_01.png")
    print("Example true star position:", info["x_true"][0], info["y_true"][0])

