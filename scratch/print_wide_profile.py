import cv2
import numpy as np
import os
import mss

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
    
    # 1D smoothing
    kernel = np.ones(5) / 5.0
    smoothed = np.convolve(profile, kernel, mode='same')
    
    # Print profile around the first few peaks (x=0 to 120)
    print("Smoothed profile values at x=0..120:")
    for x in range(0, 120):
        bar = "#" * int(smoothed[x])
        print(f"x={x:3d}: val={smoothed[x]:5.2f} | {bar}")

if __name__ == "__main__":
    main()
