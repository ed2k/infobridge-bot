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
    
    hsv = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2HSV)
    
    # Define RED mask (Hearts/Diamonds)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([25, 255, 255])
    lower_red2 = np.array([165, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Define BLACK mask (Spades/Clubs)
    # In BBO/IntoBridge, card suits are black. Black has low Value (brightness).
    # Since the card background is white, black is very distinct.
    # Let's say black is Value < 100.
    mask_black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 100]))
    
    # Combine red and black masks
    mask_suit = mask_red + mask_black
    
    # Let's only look in the suit row y = 37..50
    suit_row_mask = mask_suit[37:50, :]
    
    # Sum vertically to get a 1D horizontal profile
    profile = np.sum(suit_row_mask > 0, axis=0)
    
    print("Horizontal suit pixel profile (x=0..width):")
    # Find peaks in this profile
    peaks = []
    min_dist = 5
    for x in range(min_dist, len(profile) - min_dist):
        val = profile[x]
        # We require at least 3 pixels matching the suit color in the column
        if val >= 3:
            # Check if it's a local maximum
            is_max = True
            for dx in range(-min_dist, min_dist + 1):
                if profile[x + dx] > val:
                    is_max = False
                    break
            if is_max:
                if not peaks or (x - peaks[-1]["x"]) >= min_dist:
                    # Determine suit color at this peak
                    # Check if red or black pixels are dominant in this column
                    col_red = np.sum(mask_red[37:50, x] > 0)
                    col_black = np.sum(mask_black[37:50, x] > 0)
                    color = "RED" if col_red >= col_black else "BLACK"
                    peaks.append({"x": x, "val": val, "color": color})
                    
    print(f"\nFound {len(peaks)} card suit peaks using color-based profile:")
    for idx, p in enumerate(peaks):
        print(f"  Card {idx+1:2d} at x={p['x']:3d} | count={p['val']} | color={p['color']}")

if __name__ == "__main__":
    main()
