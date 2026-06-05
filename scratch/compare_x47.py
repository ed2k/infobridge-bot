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
    gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
    
    # Crop at x = 47 (x_center=47, x_start=41)
    crop = gray[37:50, 41:54]
    
    print("\n--- Live crop at x=41..54 (shape: {}) ---".format(crop.shape))
    for r in range(crop.shape[0]):
        row_str = " ".join([f"{val:3d}" for val in crop[r]])
        print(row_str)
        
    # Print diamond template
    dia_tpl_path = "templates/diamond.png"
    if os.path.exists(dia_tpl_path):
        dia_tpl = cv2.imread(dia_tpl_path, cv2.IMREAD_GRAYSCALE)
        print("\n--- Diamond Template (shape: {}) ---".format(dia_tpl.shape))
        for r in range(dia_tpl.shape[0]):
            row_str = " ".join([f"{val:3d}" for val in dia_tpl[r]])
            print(row_str)

if __name__ == "__main__":
    main()
