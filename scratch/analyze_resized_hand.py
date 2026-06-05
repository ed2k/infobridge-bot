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
    
    gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
    
    # For each row (0 to 59), let's calculate the average pixel value across the width
    row_means = np.mean(gray, axis=1)
    
    print("Row-by-row average brightness (0 is black, 255 is white):")
    for r in range(60):
        bar = "#" * int(row_means[r] / 5)
        print(f"Row {r:2d}: mean={row_means[r]:5.1f} | {bar}")

if __name__ == "__main__":
    main()
