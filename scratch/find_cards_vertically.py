import cv2
import numpy as np
import os
import json

def main():
    img_path = "debug_captures/1_ui_full.png"
    if not os.path.exists(img_path):
        print(f"❌ Full UI image not found at {img_path}")
        return
        
    img = cv2.imread(img_path)
    print(f"Loaded {img_path} (shape: {img.shape})")
    
    # Load config to get margins
    with open("config.json", "r") as f:
        config = json.load(f)
        
    ui_x = config["ui_roi"]["x"]
    ui_y = config["ui_roi"]["y"]
    
    # Hand coordinates relative to UI ROI
    hand_x_rel = config["player_hand_roi"]["x"] - ui_x
    hand_w = config["player_hand_roi"]["width"]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x_start = max(0, hand_x_rel)
    x_end = min(img.shape[1], hand_x_rel + hand_w)
    
    print(f"Scanning relative columns x from {x_start} to {x_end}")
    
    best_y_ranges = []
    current_range_start = None
    
    for y in range(0, img.shape[0]):
        row_slice = gray[y, x_start:x_end]
        white_ratio = np.sum(row_slice > 200) / len(row_slice) # lowered white threshold to 200 just in case
        
        # If more than 30% of the row is white, it's likely part of the hand cards
        if white_ratio > 0.30:
            if current_range_start is None:
                current_range_start = y
        else:
            if current_range_start is not None:
                best_y_ranges.append((current_range_start, y - 1))
                current_range_start = None
                
    if current_range_start is not None:
        best_y_ranges.append((current_range_start, img.shape[0] - 1))
        
    print("\nFound white regions (card bands) in UI space at relative y-coordinates:")
    for start, end in best_y_ranges:
        height = end - start + 1
        global_start = start + ui_y
        global_end = end + ui_y
        print(f"  - Relative y: {start} to {end} (height: {height}px) -> Global y: {global_start} to {global_end}")
        
    # Let's recommend the correct ROI coordinates
    if best_y_ranges:
        for start, end in best_y_ranges:
            height = end - start + 1
            if 30 <= height <= 150:
                rec_y_rel = start - 3
                rec_h = height + 6
                global_rec_y = rec_y_rel + ui_y
                print(f"\n💡 Recommended new player_hand_roi settings:")
                print(f"   \"x\": {config['player_hand_roi']['x']}")
                print(f"   \"y\": {global_rec_y}")
                print(f"   \"width\": {hand_w}")
                print(f"   \"height\": {rec_h}")
                
                crop = img[max(0, rec_y_rel):min(img.shape[0], rec_y_rel + rec_h), x_start:x_end]
                os.makedirs("debug_ocr_test", exist_ok=True)
                cv2.imwrite("debug_ocr_test/detected_hand_crop.png", crop)
                print(f"   Saved sample crop to debug_ocr_test/detected_hand_crop.png")

if __name__ == "__main__":
    main()
