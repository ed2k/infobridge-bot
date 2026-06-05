import cv2
import numpy as np
import os

def main():
    img_path = "debug_captures/4_player_hand.png"
    if not os.path.exists(img_path):
        print("❌ Hand image not found.")
        return
        
    img = cv2.imread(img_path)
    print(f"Loaded {img_path} (shape: {img.shape})")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Sum columns vertically to see the white segments
    col_sums = np.sum(thresh > 0, axis=0)
    
    # Find segments where col_sums > 0
    segments = []
    in_segment = False
    start = 0
    for x in range(len(col_sums)):
        if col_sums[x] > 5: # threshold for card presence in column
            if not in_segment:
                in_segment = True
                start = x
        else:
            if in_segment:
                in_segment = False
                segments.append((start, x - 1))
    if in_segment:
        segments.append((start, len(col_sums) - 1))
        
    print(f"\nFound {len(segments)} white segments (potential card groups):")
    for idx, (s_start, s_end) in enumerate(segments):
        w = s_end - s_start + 1
        print(f"  Group {idx+1}: x={s_start}..{s_end} (width: {w}px)")

if __name__ == "__main__":
    main()
