import cv2
import numpy as np
import os
import mss

def to_ascii(img):
    # Threshold to make it binary-like for clear ASCII visualization
    chars = []
    for y in range(img.shape[0]):
        row = []
        for x in range(img.shape[1]):
            val = img[y, x]
            if val > 200:
                row.append(" ")
            elif val > 150:
                row.append(".")
            elif val > 100:
                row.append("x")
            else:
                row.append("#")
        chars.append("".join(row))
    return chars

def main():
    monitor = {
        "top": 871,
        "left": 1170,
        "width": 520,
        "height": 65
    }
    
    with mss.mss() as sct:
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        hand_crop = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
    # Card 6 (Club Q) suit center is at 222
    # Card 7 (Club 8) suit center is at 260
    
    c6_card = hand_crop[0:60, 222-15:222+25]
    c6_suit = cv2.cvtColor(c6_card[24:55, 6:24], cv2.COLOR_BGR2GRAY)
    
    c7_card = hand_crop[0:60, 260-15:260+25]
    c7_suit = cv2.cvtColor(c7_card[24:55, 6:24], cv2.COLOR_BGR2GRAY)
    
    # We want to find the exact 13x13 bounding box of the symbol in each crop
    # Let's find the best match for the club template in each
    club_tpl = cv2.imread("templates/club.png", cv2.IMREAD_GRAYSCALE)
    spade_tpl = cv2.imread("templates/spade.png", cv2.IMREAD_GRAYSCALE)
    
    res6 = cv2.matchTemplate(c6_suit, club_tpl, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res6)
    x6, y6 = max_loc
    c6_best = c6_suit[y6:y6+13, x6:x6+13]
    
    res7 = cv2.matchTemplate(c7_suit, club_tpl, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res7)
    x7, y7 = max_loc
    c7_best = c7_suit[y7:y7+13, x7:x7+13]
    
    ascii_c6 = to_ascii(c6_best)
    ascii_c7 = to_ascii(c7_best)
    ascii_club = to_ascii(club_tpl)
    ascii_spade = to_ascii(spade_tpl)
    
    print("\n--- CLUB TEMPLATE ---")
    for r in ascii_club:
        print(r)
        
    print("\n--- SPADE TEMPLATE ---")
    for r in ascii_spade:
        print(r)
        
    print("\n--- CARD 6 (CLUB Q) BEST MATCH ---")
    for r in ascii_c6:
        print(r)
        
    print("\n--- CARD 7 (CLUB 8) BEST MATCH ---")
    for r in ascii_c7:
        print(r)

if __name__ == "__main__":
    main()
