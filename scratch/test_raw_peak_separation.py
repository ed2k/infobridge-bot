import cv2
import numpy as np
import os
import mss
from analyzer import BridgeAnalyzer

def main():
    monitor = {
        "top": 871,
        "left": 1215,
        "width": 455,
        "height": 65
    }
    
    with mss.mss() as sct:
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        hand_crop = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
    hsv = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2HSV)
    
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([25, 255, 255])
    lower_red2 = np.array([165, 40, 40])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    mask_black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 60, 100]))
    mask_suit = mask_red + mask_black
    
    profile = np.sum(mask_suit[37:50, :] > 0, axis=0).astype(np.float32)
    
    # Peak detection on RAW profile
    print("\n--- Testing Peak Detection on RAW Profile ---")
    for min_dist in [4, 5, 6]:
        peaks = []
        for x in range(min_dist, len(profile) - min_dist):
            val = profile[x]
            if val >= 3.0: # require at least 3 pixels matching color
                is_max = True
                for dx in range(-min_dist, min_dist + 1):
                    if profile[x + dx] > val:
                        is_max = False
                        break
                if is_max:
                    if not peaks or (x - peaks[-1]) >= min_dist:
                        peaks.append(x)
        print(f"min_dist={min_dist}: Found {len(peaks)} peaks at x: {peaks}")

if __name__ == "__main__":
    main()
