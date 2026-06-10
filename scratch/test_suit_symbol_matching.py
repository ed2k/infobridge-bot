import cv2
import numpy as np
from analyzer import BridgeAnalyzer

def main():
    analyzer = BridgeAnalyzer(verbose=False)
    img = cv2.imread("debug_captures/live_ui_all_sides.png")
    if img is None:
        print("Error")
        return
        
    blocks = [
        (19, 75, 114, 85, "Block 1"),
        (142, 75, 114, 85, "Block 2"),
        (279, 75, 100, 85, "Block 3")
    ]
    
    for bx, by, bw, bh, name in blocks:
        # Crop the small suit icon at the top-left of the block
        suit_crop = img[by+6 : by+18, bx+2 : bx+20]
        gray = cv2.cvtColor(suit_crop, cv2.COLOR_BGR2GRAY)
        
        # We run template matching against all 4 suits
        scores = {}
        for suit in ["spade", "heart", "diamond", "club"]:
            template = analyzer.suit_templates.get(suit)
            if template is None:
                continue
            t_h, t_w = template.shape[:2]
            g_h, g_w = gray.shape[:2]
            
            if g_h < t_h or g_w < t_w:
                gray_search = cv2.resize(gray, (max(g_w, t_w), max(g_h, t_h)))
            else:
                gray_search = gray
                
            res = cv2.matchTemplate(gray_search, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            scores[suit] = max_val
            
        best_suit = max(scores, key=scores.get)
        print(f"{name}: Best match = {best_suit} (score={scores[best_suit]:.3f})")
        for s, score in scores.items():
            print(f"  {s}: {score:.3f}")

if __name__ == "__main__":
    main()
