import cv2
import numpy as np

def main():
    img = cv2.imread("debug/player_hand_area.png")
    if img is None:
        print("❌ Failed to load image")
        return
        
    h_strip = img.shape[0]
    w_strip = img.shape[1]
    scale = 60.0 / h_strip
    hand_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    hsv = cv2.cvtColor(hand_img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([25, 255, 255])
    lower_red2 = np.array([165, 40, 40])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    mask_black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 60, 100]))
    
    peaks = [48, 84, 121, 158, 195, 231, 268, 305, 342, 378, 415, 452]
    
    print("Peak | col_red | col_black | HSV values at y=41..54")
    print("-" * 60)
    for p in peaks:
        red_cnt = np.sum(mask_red[41:54, p] > 0)
        black_cnt = np.sum(mask_black[41:54, p] > 0)
        
        # Let's see some HSV values at y=41..54
        hsv_slice = [hsv[y, p].tolist() for y in range(41, 54)]
        print(f"{p:3d}  | {red_cnt:7d} | {black_cnt:9d} | {hsv_slice[0]} ... {hsv_slice[-1]}")

if __name__ == "__main__":
    main()
