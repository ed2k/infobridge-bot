import cv2
import numpy as np

def main():
    img = cv2.imread("debug_captures/1_ui_full.png")
    if img is None:
        print("Failed to load image")
        return
        
    dummy_text_crop = img[275:310, 0:510]
    gray = cv2.cvtColor(dummy_text_crop, cv2.COLOR_BGR2GRAY)
    
    # Run Otsu thresholding to get binary image (text=0, background=255 or vice-versa)
    # Background is typically light (gray/white), text is dark.
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # We want text to be 1 (or 255) and background to be 0
    # Let's check which is more common at the borders to see if background is white
    if np.mean(thresh[:, :10]) > 127:
        thresh = cv2.bitwise_not(thresh)
        
    # Vertical projection (sum along columns)
    proj = np.sum(thresh > 0, axis=0)
    
    print("Horizontal profile of text pixel columns (sum of rows per x):")
    # Let's find vertical lines where the sum is 0 (or very close to 0)
    # These represent gaps between columns.
    gaps = []
    in_gap = False
    gap_start = 0
    for x in range(len(proj)):
        # Allow a threshold of 1 pixel to handle noise
        is_empty = (proj[x] <= 1)
        if is_empty:
            if not in_gap:
                in_gap = True
                gap_start = x
        else:
            if in_gap:
                in_gap = False
                gaps.append((gap_start, x - 1))
    if in_gap:
        gaps.append((gap_start, len(proj) - 1))
        
    print(f"Detected empty gaps between text: {gaps}")
    
    # Let's also print the projection values around x=100..160, 220..280, 340..400
    # to see where the dips are!
    for region in [(100, 160), (220, 280), (340, 400)]:
        r_start, r_end = region
        sub = proj[r_start:r_end]
        min_x = r_start + np.argmin(sub)
        print(f"Region x={r_start}..{r_end}: min projection value is {proj[min_x]} at x={min_x}")

if __name__ == "__main__":
    main()
