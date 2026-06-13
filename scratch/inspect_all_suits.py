import cv2
import numpy as np

def main():
    img = cv2.imread("debug/dummy_strip_east.png")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Red mask
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([25, 255, 255])
    lower_red2 = np.array([165, 40, 40])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Black mask
    black_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 60, 100]))

    # Positions of detected cards from template matching
    detections = [
        ("Row 1 (y=42, x=47)", 47, 42),
        ("Row 1 (y=42, x=81)", 81, 42),
        ("Row 2 (y=100, x=12)", 12, 100),
        ("Row 2 (y=94, x=46)", 46, 94),
        ("Row 2 (y=92, x=82)", 82, 92),
        ("Row 3 (y=157, x=5)", 5, 157),
        ("Row 3 (y=149, x=27)", 27, 149),
        ("Row 3 (y=157, x=47)", 47, 157),
        ("Row 3 (y=157, x=69)", 69, 157),
        ("Row 3 (y=157, x=90)", 90, 157),
        ("Row 4 (y=215, x=47)", 47, 215),
        ("Row 4 (y=215, x=81)", 81, 215)
    ]

    print("Checking suit colors at the expected suit symbol positions (sx = rx + 7, sy = ry + 28):")
    for name, rx, ry in detections:
        sx = rx + 7
        sy = ry + 28
        
        # Check a 13x13 window around the suit symbol
        crop_red = red_mask[sy:sy+13, sx:sx+13]
        crop_black = black_mask[sy:sy+13, sx:sx+13]
        
        red_sum = np.sum(crop_red > 0)
        black_sum = np.sum(crop_black > 0)
        
        color = "UNKNOWN"
        if red_sum > black_sum and red_sum > 10:
            color = "RED"
        elif black_sum > red_sum and black_sum > 10:
            color = "BLACK"
            
        print(f"  {name:<25} -> Red pixels: {red_sum:<3} | Black pixels: {black_sum:<3} | Color: {color}")

if __name__ == "__main__":
    main()
