import cv2
import numpy as np

def analyze_block_peaks(img, bx, by, bw, bh, name):
    gray = cv2.cvtColor(img[by:by+bh, bx:bx+bw], cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # We look at the top part of the card block where ranks are (y_off = 2..28)
    rank_strip = thresh[2:28, :]
    
    # Calculate column-wise sum of black pixels (text/borders)
    col_sums = np.sum(rank_strip == 0, axis=0)
    
    print(f"\n--- {name} Profile (w={bw}) ---")
    # Let's print out the column sums in a readable way to see where the transitions are
    # A card transition/border typically has a vertical dark line (high column sum of black pixels)
    # Ranks also have high column sums.
    # We can print indices where the sum is high
    for x in range(bw):
        if col_sums[x] > 5:
            print(f"  x={x:3d}: sum={col_sums[x]:2d}")

def main():
    img = cv2.imread("debug_captures/live_ui_all_sides.png")
    if img is None:
        print("Error")
        return
        
    analyze_block_peaks(img, 19, 75, 114, 85, "Block 1 (Spade)")
    analyze_block_peaks(img, 142, 75, 114, 85, "Block 2 (Heart)")
    analyze_block_peaks(img, 265, 75, 114, 85, "Block 7 (Diamond)")

if __name__ == "__main__":
    main()
