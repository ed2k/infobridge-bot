import cv2
import numpy as np
import os
import mss
import pytesseract
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
        
    analyzer = BridgeAnalyzer(verbose=False)
    
    hsv = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2HSV)
    
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([25, 255, 255])
    lower_red2 = np.array([165, 40, 40])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    mask_black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 60, 100]))
    mask_suit = mask_red + mask_black
    
    profile = np.sum(mask_suit[37:50, :] > 0, axis=0).astype(np.float32)
    
    # Peak detection with min_dist=5
    min_dist = 5
    peaks = []
    for x in range(min_dist, len(profile) - min_dist):
        val = profile[x]
        if val >= 3.0:
            is_max = True
            for dx in range(-min_dist, min_dist + 1):
                if profile[x + dx] > val:
                    is_max = False
                    break
            if is_max:
                if not peaks or (x - peaks[-1]["x"]) >= min_dist:
                    col_red = np.sum(mask_red[37:50, x] > 0)
                    col_black = np.sum(mask_black[37:50, x] > 0)
                    color = "RED" if col_red >= col_black else "BLACK"
                    peaks.append({"x": x, "color": color})
                    
    print(f"\nAnalyzing {len(peaks)} peaks:")
    for idx, p in enumerate(peaks):
        x_val = p["x"]
        x_card = max(0, x_val - 5)
        card_crop = hand_crop[0:60, x_card:min(x_card + 40, hand_crop.shape[1])]
        
        # OCR rank
        rank, suit = analyzer.extract_card(card_crop)
        if not suit:
            suit = "heart" if p["color"] == "RED" else "spade"
            
        print(f"  Peak {idx+1:2d} at x={x_val:3d} | color={p['color']} | Rank={rank or '?'} | Suit={suit}")

if __name__ == "__main__":
    main()
