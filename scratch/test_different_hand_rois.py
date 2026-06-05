import cv2
import numpy as np
import os
from analyzer import BridgeAnalyzer

def main():
    img_path = "debug_captures/1_ui_full.png"
    if not os.path.exists(img_path):
        print(f"❌ Full UI image not found at {img_path}")
        return
        
    img = cv2.imread(img_path)
    print(f"Loaded {img_path} (shape: {img.shape})")
    
    analyzer = BridgeAnalyzer(verbose=True)
    
    # Let's define the candidate crops relative to 1_ui_full.png (y ranges)
    # x ranges: hand_x_rel = 1260 - 1245 = 15, width = 418. So x from 15 to 433.
    x1, x2 = 15, 433
    
    candidates = [
        ("Band 1 (Relative y: 76-106)", 76, 106),
        ("Band 2 (Relative y: 267-331)", 267, 331),
        ("Band 3 (Relative y: 384-503)", 384, 503),
        ("Band 4 (Relative y: 517-546)", 517, 546),
        ("Band 5 (Relative y: 552-580)", 552, 580),
        ("Band 6 (Relative y: 651-716)", 651, 716)
    ]
    
    for name, y1, y2 in candidates:
        crop = img[y1:y2, x1:x2]
        cv2.imwrite(f"debug_ocr_test/crop_{y1}_{y2}.png", crop)
        
        # Test detection
        print(f"\n--- Testing {name} (crop shape: {crop.shape}) ---")
        try:
            detected_hand = analyzer.extract_hand_cards(crop)
            if detected_hand:
                cards_str = [f"{c['rank'] or '?'}{c['suit'] or '?'}" for c in detected_hand]
                print(f"  ✅ Detected {len(detected_hand)} cards:")
                print("  " + ", ".join(cards_str))
            else:
                print("  ❌ No cards detected.")
        except Exception as e:
            print(f"  ❌ Error during detection: {e}")

if __name__ == "__main__":
    main()
