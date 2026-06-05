import cv2
import numpy as np
import os
import mss
from analyzer import BridgeAnalyzer

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
        
    analyzer = BridgeAnalyzer(verbose=False)
    
    # Card 6 (Club Q) peak is at 222
    # Card 7 (Club 8) peak is at 260
    for name, p_x in [("Card 6 (Club Q)", 222), ("Card 7 (Club 8)", 260)]:
        print(f"\n=== Analyzing {name} at peak {p_x} ===")
        # Let's test different center shifts from -3 to +3
        for shift in range(-3, 4):
            x_center = p_x + shift
            x_card = max(0, x_center - 15)
            card_crop = hand_crop[0:60, x_card:min(x_card + 40, hand_crop.shape[1])]
            suit_crop = card_crop[24:55, 6:24]
            gray = cv2.cvtColor(suit_crop, cv2.COLOR_BGR2GRAY)
            
            # Match templates
            scores = {}
            for s in ["spade", "club"]:
                template = analyzer.suit_templates.get(s)
                if template is not None:
                    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    scores[s] = max_val
            print(f"  Shift {shift:+d} (x_center={x_center}): spade={scores['spade']:.3f}, club={scores['club']:.3f} | Diff={scores['club']-scores['spade']:.3f}")

if __name__ == "__main__":
    main()
