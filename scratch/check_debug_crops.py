import cv2
import numpy as np
import os

def main():
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

    suits = ["spade", "heart", "diamond", "club"]
    suit_templates = {}
    for s in suits:
        p = os.path.join(templates_dir, f"{s}.png")
        if os.path.exists(p):
            suit_templates[s] = cv2.imread(p, cv2.IMREAD_GRAYSCALE)

    print("Analyzing dummy_card_East_* crops in debug/ using template matching:")
    for filename in sorted(os.listdir("debug")):
        if filename.startswith("dummy_card_East_"):
            filepath = os.path.join("debug", filename)
            img = cv2.imread(filepath)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Match ranks
            best_r = None
            best_r_score = -1
            for r, tpl in rank_templates.items():
                res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val > best_r_score:
                    best_r_score = max_val
                    best_r = r
                    
            # Match suits
            best_s = None
            best_s_score = -1
            for s, tpl in suit_templates.items():
                res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val > best_s_score:
                    best_s_score = max_val
                    best_s = s
                    
            print(f"  {filename:<35} -> Matched: {best_r}{best_s[0].upper()} (Rank score: {best_r_score:.3f}, Suit score: {best_s_score:.3f})")

if __name__ == "__main__":
    main()
