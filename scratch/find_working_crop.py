import cv2
import os
import json
from analyzer import BridgeAnalyzer

def main():
    ui_img_path = "/Users/admin/Documents/GitHub/infobridge-bot/debug_captures/1_ui_full.png"
    if not os.path.exists(ui_img_path):
        print("Error: 1_ui_full.png not found")
        return
        
    ui_img = cv2.imread(ui_img_path)
    analyzer = BridgeAnalyzer(verbose=True)
    
    # We will try different absolute coordinates (relative to ui_roi: x=1245, y=220)
    # We'll test rel_y from 130 to 150, and heights from 95 to 125
    test_cases = [
        {"y": 355, "height": 120}, # Old default
        {"y": 358, "height": 115},
        {"y": 360, "height": 110},
        {"y": 362, "height": 108},
        {"y": 365, "height": 105},
        {"y": 368, "height": 102},
        {"y": 370, "height": 100},
        {"y": 372, "height": 98},
    ]
    
    for tc in test_cases:
        y = tc["y"]
        h = tc["height"]
        rel_y = y - 220
        rel_x = 106
        w = 210
        
        crop = ui_img[rel_y:rel_y+h, rel_x:rel_x+w]
        print(f"\n==========================================")
        print(f"Testing absolute y={y} (rel_y={rel_y}), height={h}")
        print(f"==========================================")
        
        # Test extraction
        try:
            bids = analyzer.extract_bids(crop)
            if bids:
                print(f"✅ SUCCESS! Detected {len(bids)} bids:")
                print(" -> ".join([f"{direction}:{bid}" for direction, bid in bids]))
            else:
                print("❌ Failed: No bids/headers detected.")
        except Exception as e:
            print(f"💥 Exception: {e}")

if __name__ == "__main__":
    main()
