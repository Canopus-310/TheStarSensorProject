
import csv, math, numpy as np, os
detf = "C:/Users/aryan/PycharmProjects/TheStarSensorProject/results/detected_centroids.csv"
truthf = "C:/Users/aryan/PycharmProjects/TheStarSensorProject/results/catalog_sky_01_truth.csv"
det = []
with open(detf) as f:
    r = csv.DictReader(f)
    for row in r:
        det.append((float(row['x']), float(row['y'])))
truth=[]
with open(truthf) as f:
    r = csv.DictReader(f)
    for row in r:
        truth.append((float(row['u']), float(row['v'])))
det = np.array(det)
truth = np.array(truth)
# quick nearest-neighbor (bruteforce) to estimate mean dx,dy for matches within 8px
from math import hypot
matches = []
for dx,dy in det:
    dists = np.hypot(truth[:,0]-dx, truth[:,1]-dy)
    i = dists.argmin()
    if dists[i] <= 8:
        matches.append((dx-truth[i,0], dy-truth[i,1], dists[i]))
if not matches:
    print("No near matches found (threshold 8 px).")
else:
    arr = np.array(matches)
    print("Matched:", len(matches))
    print("mean dx,dy:", arr[:,0].mean(), arr[:,1].mean())
    print("median dist:", np.median(arr[:,2]))
    print("max dist:", arr[:,2].max())
    # save dx,dy histogram stats
    np.savetxt("results/match_offsets.csv", arr, header="dx,dy,dist", delimiter=",")
    print("Saved results/match_offsets.csv")
