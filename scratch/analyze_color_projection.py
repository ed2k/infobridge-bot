import cv2
import numpy as np

def main():
    img = cv2.imread("debug_captures/1_ui_full.png")
    if img is None:
        print("Failed to load image")
        return
        
    crop = img[275:310, 0:510]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    # Define RED mask (Hearts/Diamonds)
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([25, 255, 255])
    lower_red2 = np.array([165, 40, 40])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Define BLACK mask (Spades/Clubs)
    mask_black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 60, 200]))
    
    # Define GREEN/CLUB color if BBO uses green for clubs (under alternating colors)
    # Green is roughly hue 35..85
    mask_green = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    
    # Project vertically
    red_proj = np.sum(mask_red > 0, axis=0)
    black_proj = np.sum(mask_black > 0, axis=0)
    green_proj = np.sum(mask_green > 0, axis=0)
    
    # Total projection of text
    total_proj = red_proj + black_proj + green_proj
    
    print("Color horizontal projection (chunked every 10px):")
    for i in range(0, 510, 10):
        chunk_total = total_proj[i:i+10]
        chunk_red = red_proj[i:i+10]
        chunk_black = black_proj[i:i+10]
        chunk_green = green_proj[i:i+10]
        
        # Build colored visualization
        bar = []
        for x in range(i, min(i+10, 510)):
            if total_proj[x] <= 1:
                bar.append(".")
            elif red_proj[x] > max(black_proj[x], green_proj[x]):
                bar.append("R")
            elif green_proj[x] > black_proj[x]:
                bar.append("G")
            else:
                bar.append("B")
        bar_str = "".join(bar)
        
        print(f"x={i:3d}..{i+9:3d}: {bar_str} | Red={list(map(int, chunk_red))} Black={list(map(int, chunk_black))} Green={list(map(int, chunk_green))}")

if __name__ == "__main__":
    main()
