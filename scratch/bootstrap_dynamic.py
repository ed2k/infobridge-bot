import cv2
import numpy as np
import os
import mss

def main():
    # Capture the wide hand crop: x = 1170, y = 871, width = 520, height = 65
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
        
    # Resize to height 60 (matching normalization in analyzer.py)
    scale = 60.0 / hand_crop.shape[0]
    hand_crop = cv2.resize(hand_crop, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    os.makedirs("templates", exist_ok=True)
    
    # Exact center peak locations of the suits in the scaled 60-height image:
    # Spade: global 1198 (relative 28 -> scaled 26)
    # Heart: global 1276 (relative 106 -> scaled 98)
    # Club: global 1392 (relative 222 -> scaled 205)
    # Diamond: global 1468 (relative 298 -> scaled 275)
    targets = {
        "spade": 26,
        "heart": 98,
        "club": 205,
        "diamond": 275
    }
    
    print("✂️ Extracting dynamic suit templates...")
    for suit, x_center in targets.items():
        # Crop 13x13 centered at x_center
        x_start = x_center - 6
        x_end = x_center + 7
        y_start = 41
        y_end = 54
        
        crop = hand_crop[y_start:y_end, x_start:x_end]
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        out_path = os.path.join("templates", f"{suit}.png")
        cv2.imwrite(out_path, gray_crop)
        print(f"   Saved templates/{suit}.png (Size: {gray_crop.shape}, center={x_center})")
        
    print("\n🎉 Dynamic templates bootstrapped successfully!")

if __name__ == "__main__":
    main()
