import cv2
import numpy as np
from analyzer import BridgeAnalyzer

def main():
    analyzer = BridgeAnalyzer(verbose=True)
    img = cv2.imread("debug_captures/live_ui_all_sides.png")
    if img is None:
        print("Error: debug_captures/live_ui_all_sides.png not found")
        return
        
    blocks = [
        (19, 75, 114, 85),
        (142, 75, 114, 85),
        (279, 75, 100, 85)
    ]
    
    for idx, (bx, by, bw, bh) in enumerate(blocks):
        print(f"\n--- Block {idx+1}: x={bx}..{bx+bw}, y={by}..{by+bh} ---")
        
        # Crop the suit region of the first card
        # Let's try multiple vertical offsets for suit
        for y_off in [24, 28, 30, 32]:
            suit_crop = img[by+y_off : by+y_off+26, bx+4 : bx+22]
            
            # Check HSV red ratio
            hsv = cv2.cvtColor(suit_crop, cv2.COLOR_BGR2HSV)
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([25, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_ratio = np.sum((mask1 + mask2) > 0) / suit_crop.size
            
            suit_tm = analyzer.classify_suit_template_matching(suit_crop)
            suit_cs = analyzer.classify_suit_by_color_shape(suit_crop)
            print(f"  y_off={y_off}: red_ratio={red_ratio:.3f}, template_matching={suit_tm}, color_shape={suit_cs}")

if __name__ == "__main__":
    main()
