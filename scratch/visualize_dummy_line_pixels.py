import cv2
import numpy as np

def main():
    img = cv2.imread("debug_captures/live_ui_all_sides.png")
    if img is None:
        print("Error")
        return
        
    crop = img[278:307, 0:510]
    h, w = crop.shape[:2]
    
    # We want to find which x coordinates have red, green, black, etc.
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    # Red mask (for Hearts / Diamonds)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([165, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Green mask (for Clubs if 4-color)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Print horizontal projection of red and green pixels
    red_proj = np.sum(red_mask > 0, axis=0)
    green_proj = np.sum(green_mask > 0, axis=0)
    
    print("Red pixel counts horizontally (x=0..510):")
    for x in range(w):
        if red_proj[x] > 0:
            print(f"  x={x}: red={red_proj[x]}")
            
    print("\nGreen pixel counts horizontally (x=0..510):")
    for x in range(w):
        if green_proj[x] > 0:
            print(f"  x={x}: green={green_proj[x]}")

if __name__ == "__main__":
    main()
