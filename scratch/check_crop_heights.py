import cv2
import numpy as np
import os

def main():
    img_path = "debug_ocr_test/visualized_peaks.png"
    if not os.path.exists(img_path):
        print("❌ Visualized image not found.")
        return
        
    img = cv2.imread(img_path)
    print(f"Loaded visualized image (shape: {img.shape})")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Let's print the average brightness of each row from y=0 to 65
    print("\nRow-by-row average brightness in crop:")
    for r in range(img.shape[0]):
        row_mean = np.mean(gray[r, :])
        bar = "#" * int(row_mean / 5)
        print(f"Row {r:2d}: mean={row_mean:5.1f} | {bar}")

if __name__ == "__main__":
    main()
