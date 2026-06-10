import cv2
import numpy as np

def main():
    img = cv2.imread("debug_captures/live_ui_all_sides.png")
    if img is None:
        print("Error")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # We crop the top card region
    crop = gray[75:160, 0:400]
    h, w = crop.shape[:2]
    
    # We map 2x2 blocks of pixels to a single character to fit in terminal
    chars = " .:-=+*#%@"
    for r in range(0, h, 2):
        row_str = ""
        for c in range(0, w, 2):
            block = crop[r:r+2, c:c+2]
            val = np.mean(block)
            idx = int(val / 256.0 * len(chars))
            row_str += chars[idx]
        print(row_str)

if __name__ == "__main__":
    main()
