import cv2
import numpy as np

def main():
    img = cv2.imread("debug/player_hand_area.png")
    if img is None:
        print("❌ Failed to load image")
        return
        
    print(f"Image shape: {img.shape}")
    
    # Save the left 60 pixels of the original crop
    left_crop = img[:, 0:60]
    cv2.imwrite("debug/inspect_extreme_left.png", left_crop)
    print("Saved debug/inspect_extreme_left.png")
    
    # Print pixel values of leftmost column to check if it has a border or is cut off
    # We look at HSV values at y=30 (middle of the crop)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    print("HSV values at y=30 for x=0..20:")
    for x in range(20):
        print(f"  x={x:2d}: {hsv[30, x].tolist()}")

if __name__ == "__main__":
    main()
