import cv2
import numpy as np
import os

def main():
    img = cv2.imread("debug/dummy_strip_east.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    templates_dir = "templates"
    ranks = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
    rank_templates = {}
    for r in ranks:
        p = os.path.join(templates_dir, f"rank_{r}.png")
        if os.path.exists(p):
            rank_templates[r] = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        else:
            p2 = os.path.join(templates_dir, f"{r}.png")
            if os.path.exists(p2):
                rank_templates[r] = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)

    # We crop the rank area for the two cards in Row 4:
    # Card 1: x_rank = 47, y_rank = 215. Crop region: y=215..241, x=47..65
    # Card 2: x_rank = 81, y_rank = 215. Crop region: y=215..241, x=81..99
    
    for name, rx, ry in [("Bottom Card 1 (x=47)", 47, 215),
                         ("Bottom Card 2 (x=81)", 81, 215)]:
        rank_crop = gray[ry:ry+26, rx:rx+18]
        
        scores = []
        for r, tpl in rank_templates.items():
            res = cv2.matchTemplate(rank_crop, tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            scores.append((r, max_val))
            
        scores.sort(key=lambda x: x[1], reverse=True)
        print(f"\n{name} top matching ranks:")
        for r, score in scores[:5]:
            print(f"  {r}: score = {score:.3f}")

if __name__ == "__main__":
    main()
