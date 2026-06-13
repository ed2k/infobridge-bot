import cv2
import numpy as np
import os

def main():
    img = cv2.imread("debug/dummy_strip_east.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Let's crop suit symbols for the three cards in row 2:
    # 1. x=12, y=100 -> suit around x=20, y=128
    # 2. x=46, y=94  -> suit around x=53, y=127
    # 3. x=82, y=92  -> suit around x=88, y=127
    
    analyzer_dir = "templates"
    spade_tpl = cv2.imread(os.path.join(analyzer_dir, "spade.png"), cv2.IMREAD_GRAYSCALE)
    club_tpl = cv2.imread(os.path.join(analyzer_dir, "club.png"), cv2.IMREAD_GRAYSCALE)
    
    for name, sx, sy in [("Card 1 (x=12)", 20, 128),
                         ("Card 2 (x=46)", 53, 127),
                         ("Card 3 (x=82)", 88, 127)]:
        suit_crop = gray[sy:sy+13, sx:sx+13]
        
        # Match spade
        res_s = cv2.matchTemplate(suit_crop, spade_tpl, cv2.TM_CCOEFF_NORMED)
        _, max_s, _, _ = cv2.minMaxLoc(res_s)
        
        # Match club
        res_c = cv2.matchTemplate(suit_crop, club_tpl, cv2.TM_CCOEFF_NORMED)
        _, max_c, _, _ = cv2.minMaxLoc(res_c)
        
        print(f"{name} at ({sx}, {sy}): Spade score = {max_s:.3f}, Club score = {max_c:.3f}")

if __name__ == "__main__":
    main()
