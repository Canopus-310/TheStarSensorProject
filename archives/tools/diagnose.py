import os
import cv2
import numpy as np

# Compute project root dynamically
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# REAL results_archived_2 folder
IMG_PATH = os.path.join(PROJECT_ROOT, "results_archived_2", "catalog_sky_01_linear16.png")

print("ABS PATH:", IMG_PATH)
print("Exists:", os.path.exists(IMG_PATH))

if not os.path.exists(IMG_PATH):
    print("File not found. Re-run generator: python src/vision/catalog_based_generator.py")
    exit(1)

img = cv2.imread(IMG_PATH, -1).astype(float)

print("Shape:", img.shape)
print("dtype:", img.dtype)
print("min:", img.min())
print("max:", img.max())
print("mean:", img.mean())
