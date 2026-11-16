"""Measure each label of a segmentation result in milliliters and write volumes.csv.

Reads a downloaded result with nibabel: no key, no request. Labels are numbers here; their
names are on the model's page at nalvera.ai. Run: python organ_volumes_from_result.py
"""
import csv

import nibabel as nib
import numpy as np

RESULT = "results/ct_scan__totalsegmentator_ct__total.nii.gz"

image = nib.load(RESULT)
labels = np.asarray(image.dataobj).astype(np.int64)
voxel_ml = float(np.prod(image.header.get_zooms()[:3])) / 1000.0     # mm³ to ml
values, counts = np.unique(labels, return_counts=True)
with open("volumes.csv", "w", newline="", encoding="utf-8") as fh:
    writer = csv.writer(fh)
    writer.writerow(["label", "voxels", "ml"])
    for value, count in zip(values, counts):
        if value != 0:                                                 # 0 is the background
            writer.writerow([int(value), int(count), round(count * voxel_ml, 3)])
            print(f"label {value}: {count * voxel_ml:.1f} ml")
