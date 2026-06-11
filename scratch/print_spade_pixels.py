import cv2
import numpy as np

def main():
    img = cv2.imread("debug_captures/live_ui_all_sides.png")
    if img is None:
        print("Failed to load image")
        return
        
    crop = img[275:310, 0:510]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # Let's inspect the minimum and average values in different columns
    # Col 1: Spades (x=19..60)
    col1 = gray[:, 19:60]
    print(f"Col 1 (Spades): min={np.min(col1)}, mean={np.mean(col1)}")
    # Print the coordinates and values of pixels that are dark (e.g. < 240)
    for y in range(col1.shape[0]):
        row_vals = [col1[y, x] for x in range(col1.shape[1])]
        # print if there is any dark pixel
        if any(v < 240 for v in row_vals):
            print(f"  Row {y}: {[int(v) for v in row_vals]}")

if __name__ == "__main__":
    main()
