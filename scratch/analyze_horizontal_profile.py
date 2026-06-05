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
    
    # Let's average y from 37 to 50 for each x
    col_means = np.mean(gray[37:50, :], axis=0)
    
    print(f"Resized hand crop width: {gray.shape[1]}px")
    print("\nHorizontal brightness profile (x=0..width):")
    
    # Let's find columns where brightness drops significantly (e.g. mean < 240)
    for x in range(0, gray.shape[1]):
        if col_means[x] < 248 or x % 10 == 0:
            bar = "#" * int(col_means[x] / 10)
            print(f"x={x:3d}: mean={col_means[x]:5.1f} | {bar}")

if __name__ == "__main__":
    main()
