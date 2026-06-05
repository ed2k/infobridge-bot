import cv2
import numpy as np
import os
import mss

def test_smoothing(hand_crop, kernel_size, min_dist=15):
    hsv = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([25, 255, 255])
    lower_red2 = np.array([165, 40, 40])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    mask_black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 60, 100]))
    mask_suit = mask_red + mask_black
    
    profile = np.sum(mask_suit[37:50, :] > 0, axis=0).astype(np.float32)
    
    # Apply convolve
    kernel = np.ones(kernel_size) / float(kernel_size)
    smoothed = np.convolve(profile, kernel, mode='same')
    
    peaks = []
    for x in range(min_dist, len(smoothed) - min_dist):
        val = smoothed[x]
        if val >= 2.0:
            is_max = True
            for dx in range(-min_dist, min_dist + 1):
                if smoothed[x + dx] > val:
                    is_max = False
                    break
            if is_max:
                if not peaks or (x - peaks[-1]["x_suit"]) >= min_dist:
                    peaks.append({"x_suit": x})
                    
    print(f"\n--- Kernel Size = {kernel_size} (min_dist={min_dist}) ---")
    print(f"Found {len(peaks)} peaks:")
    p_locs = [1170 + p["x_suit"] for p in peaks]
    print("  Global coords:", p_locs)
    diffs = [p_locs[i] - p_locs[i-1] for i in range(1, len(p_locs))]
    print("  Differences:  ", diffs)
    return len(peaks)

def main():
    monitor = {
        "top": 871,
        "left": 1170,
        "width": 520,
        "height": 65
    }
    
    with mss.mss() as sct:
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        hand_crop = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
    for ks in [5, 7, 9, 11, 13, 15]:
        test_smoothing(hand_crop, kernel_size=ks, min_dist=15)

if __name__ == "__main__":
    main()
