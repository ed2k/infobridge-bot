#!/usr/bin/env python3
import cv2
import os

for s in ["spade", "heart", "diamond", "club"]:
    path = f"templates/{s}.png"
    if os.path.exists(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        print(f"Template {s}: shape={img.shape}, min_val={img.min()}, max_val={img.max()}, mean={img.mean()}")
        # Print a small ascii representation of the first template row/values
        print("First few pixels:", img[0:3, 0:5])
