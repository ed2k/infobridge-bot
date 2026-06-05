import cv2
import numpy as np
import os

def main():
    img_path = "debug_captures/1_ui_full.png"
    if not os.path.exists(img_path):
        print("❌ UI image not found.")
        return
        
    img = cv2.imread(img_path)
    hand_crop = img[651:716, 15:433]
    h_strip = hand_crop.shape[0]
    
    scale = 60.0 / h_strip
    hand_crop = cv2.resize(hand_crop, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    hsv = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2HSV)
    
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([25, 255, 255])
    lower_red2 = np.array([165, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    mask_black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 100]))
    mask_suit = mask_red + mask_black
    
    # 1D profile
    profile = np.sum(mask_suit[37:50, :] > 0, axis=0)
    
    print("Profile values at x=0..60:")
    for x in range(0, 60):
        print(f"x={x:2d}: val={profile[x]:2d}")

if __name__ == "__main__":
    main()
