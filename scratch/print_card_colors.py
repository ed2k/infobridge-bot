import cv2
import numpy as np
import os

def main():
    img_path = "debug_captures/1_ui_full.png"
    if not os.path.exists(img_path):
        print("❌ UI image not found.")
        return
        
    img = cv2.imread(img_path)
    
    # Crop Band 6: relative y 651 to 716, x 15 to 433
    hand_crop = img[651:716, 15:433]
    h_strip = hand_crop.shape[0]
    
    # Scale to height 60
    scale = 60.0 / h_strip
    hand_crop = cv2.resize(hand_crop, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    print("Resized hand crop shape:", hand_crop.shape)
    
    # Let's print out the BGR colors at y=45 (center of suit row) every 10 pixels
    print("\nBGR values at y=45 across width:")
    for x in range(0, hand_crop.shape[1], 10):
        b, g, r = hand_crop[45, x]
        print(f"x={x:3d}: BGR=({b:3d}, {g:3d}, {r:3d})")

if __name__ == "__main__":
    main()
