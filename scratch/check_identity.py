import cv2
import numpy as np

def main():
    img = cv2.imread("debug_captures/live_ui_all_sides.png")
    if img is None:
        print("Error")
        return
        
    crop1 = img[75:160, 19:133]
    crop2 = img[75:160, 142:256]
    
    diff = cv2.absdiff(crop1, crop2)
    mean_diff = np.mean(diff)
    print(f"Mean absolute pixel difference between Block 1 and Block 2: {mean_diff:.6f}")
    
    # Also check Block 3
    crop3 = img[75:160, 279:379]
    crop1_sub = img[75:160, 19:119]
    diff_3 = cv2.absdiff(crop1_sub, crop3)
    print(f"Mean absolute pixel difference between Block 1 (sub) and Block 3: {np.mean(diff_3):.6f}")

if __name__ == "__main__":
    main()
