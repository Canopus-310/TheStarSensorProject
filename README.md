# TheStarSensorProject
## Star Sensor Attitude Determination System (Ongoing)
This project corresponds to the “Star Sensor Attitude Determination System” listed under Key Projects in my CV.

This repository implements a comprehensive **star tracker pipeline** for spacecraft attitude estimation, beginning with an inertial star catalog and culminating in camera-frame attitude estimation.

### Pipeline Overview
1. **Catalog Preprocessing**
   - Converts RA/DEC star catalog to inertial-frame unit vectors  
   - File: `src/attitude/prepare_catalog_from_kaggle.py`
   - Output: `data/catalog_unit_vectors.csv`

2. **Camera Projection**
   - Projects inertial star vectors into camera pixel coordinates using a pinhole camera model
   - File: `src/scripts/project_catalog_to_pixels_point_at_bright.py`
   - Output: `catalog_projected.csv`, `R_true.txt`

3. **Synthetic Starfield Generation**
   - Generates realistic star images with PSF blur and noise
   - File: `src/vision/catalog_based_generator.py`
   - Output: `catalog_sky_01_linear16.png`, truth CSV

4. **Star Detection**
   - Detects stars using thresholding and connected-component centroiding
   - File: `src/vision/detector.py`
   - Output: `detected_centroids.csv`

5. **Detection–Truth Matching**
   - Matches detected stars to ground truth using nearest-neighbor matching
   - Files:
     - `src/scripts/match_detections_to_truth.py`
     - `src/scripts/match_refine.py`
   - Output: `match_results_refined.csv`

6. **Attitude Estimation**
   - Estimates spacecraft attitude using Wahba’s problem (SVD) with RANSAC-based outlier rejection
   - File: `src/attitude/estimate_attitude.py`
   - Output: Estimated rotation matrix and attitude error (arcseconds)

### How to Run
```bash
python src/pipeline/startracker_pipeline.py
python -m src.scripts.run_attitude_solver


Sample Results
Typical attitude error: ~1000–1500 arcsec (software-only, noise-limited)
Error sources analyzed: centroid noise, camera intrinsics, star geometry

Current Status: - 
Software pipeline complete
Ongoing work: IMU–star tracker fusion and hardware validation
