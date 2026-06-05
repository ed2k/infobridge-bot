import cv2
import numpy as np
import os
import mss
from analyzer import BridgeAnalyzer

def detect_peaks(hand_crop, min_dist=15):
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
                    col_red = np.sum(mask_red[37:50, x] > 0)
                    col_black = np.sum(mask_black[37:50, x] > 0)
                    color = "RED" if col_red >= col_black else "BLACK"
                    peaks.append({"x_suit": x, "color": color})
    return peaks

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
        
    peaks = detect_peaks(hand_crop, min_dist=15)
    os.makedirs("debug_test_crops", exist_ok=True)
    
    for idx, p in enumerate(peaks):
        x_card = max(0, p["x_suit"] - 15)
        card_crop = hand_crop[0:60, x_card:min(x_card + 40, hand_crop.shape[1])]
        
        # Crop suit area
        suit_crop = card_crop[24:55, 6:24]
        gray = cv2.cvtColor(suit_crop, cv2.COLOR_BGR2GRAY)
        
        cv2.imwrite(f"debug_test_crops/card_{idx+1}_suit.png", gray)
        print(f"Card {idx+1} (peak {p['x_suit']}): saved debug_test_crops/card_{idx+1}_suit.png, size={gray.shape}")

if __name__ == "__main__":
    main()
