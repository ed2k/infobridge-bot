import cv2
import numpy as np
import os
import mss
from analyzer import BridgeAnalyzer

def main():
    # Test a wider hand ROI
    # x = 1215, y = 871, width = 455, height = 65
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
        
    print(f"Captured wide hand crop (shape: {hand_crop.shape})")
    
    analyzer = BridgeAnalyzer(verbose=True)
    detected_hand = analyzer.extract_hand_cards(hand_crop)
    
    print("\n--- DETECTED HAND (WIDE ROI) ---")
    if detected_hand:
        cards_str = [f"{c['rank'] or '?'}{c['suit'] or '?'}" for c in detected_hand]
        print(f"Detected {len(detected_hand)} cards:")
        print(", ".join(cards_str))
    else:
        print("No cards detected.")

if __name__ == "__main__":
    main()
