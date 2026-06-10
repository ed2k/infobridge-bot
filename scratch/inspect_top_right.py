import cv2
import numpy as np

def main():
    img = cv2.imread("debug_captures/live_ui_all_sides.png")
    if img is None:
        print("Error")
        return
        
    # Crop the top-right quadrant where the 4th suit should be
    crop = img[70:170, 350:510]
    h, w = crop.shape[:2]
    
    # Print a grid of average intensities
    print("Average intensities in 10x10 grid:")
    for r in range(0, h, 10):
        row_str = ""
        for c in range(0, w, 10):
            block = crop[r:r+10, c:c+10]
            val = np.mean(block)
            row_str += f"{int(val):4d}"
        print(row_str)

    # Let's check thresholding in this crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    for thresh_val in [127, 150, 180, 200]:
        _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        white_pixels = np.sum(thresh == 255)
        print(f"Threshold {thresh_val}: {white_pixels} white pixels out of {w*h}")

if __name__ == "__main__":
    main()
