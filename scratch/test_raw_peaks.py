import cv2
import numpy as np
import os

def main():
    img_path = "debug_captures/1_ui_full.png"
    if not os.path.exists(img_path):
        print("❌ UI image not found.")
        return
        
    img = cv2.imread(img_path)
    hand_crop = img[651:716, 15:433]
    h_strip = hand_crop.shape[0]
    
    scale = 60.0 / h_strip
    hand_crop = cv2.resize(hand_crop, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    hsv = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2HSV)
    
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([25, 255, 255])
    lower_red2 = np.array([165, 40, 40])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    mask_black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 60, 100]))
    mask_suit = mask_red + mask_black
    
    profile = np.sum(mask_suit[37:50, :] > 0, axis=0).astype(np.float32)
    
    kernel = np.ones(5) / 5.0
    smoothed = np.convolve(profile, kernel, mode='same')
    
    print("\n--- ALL LOCAL MAXIMA IN SMOOTHED PROFILE ---")
    peaks_all = []
    # Test min_dist from 1 to 10
    for min_dist in [3, 4, 5, 6]:
        peaks = []
        for x in range(min_dist, len(smoothed) - min_dist):
            val = smoothed[x]
            if val > 0.5: # very low threshold
                is_max = True
                for dx in range(-min_dist, min_dist + 1):
                    if smoothed[x + dx] > val:
                        is_max = False
                        break
                if is_max:
                    if not peaks or (x - peaks[-1]) >= min_dist:
                        peaks.append(x)
        print(f"min_dist={min_dist}: Found {len(peaks)} peaks at x: {peaks}")

if __name__ == "__main__":
    main()
