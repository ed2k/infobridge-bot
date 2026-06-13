import cv2
import numpy as np

def main():
    img = cv2.imread("debug/dummy_strip_east.png")
    # Bottom area around y=210 to 250
    h, w, _ = img.shape
    
    # We want to check colors at the bottom area for x in [10..100]
    # Spades and clubs are black. Hearts and diamonds are red.
    # Let's check the red vs black ink distribution in rows y=210..250.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([25, 255, 255])
    lower_red2 = np.array([165, 40, 40])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    
    black_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 60, 100]))
    
    print("Bottom Area Color Inspection:")
    for card_name, x_range in [("Left Card (x=10..30)", (10, 30)), 
                               ("Mid-Left Card (x=35..60)", (35, 60)), 
                               ("Mid-Right Card (x=70..95)", (70, 95))]:
        red_px = np.sum(red_mask[210:, x_range[0]:x_range[1]] > 0)
        black_px = np.sum(black_mask[210:, x_range[0]:x_range[1]] > 0)
        print(f"  {card_name}: Red pixels = {red_px}, Black pixels = {black_px}")

if __name__ == "__main__":
    main()
