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
        
        # Test a range of coordinates for the suit crop of the first card
        for y_off in [24, 28, 32]:
            for x_off in [4, 6, 8]:
                suit_crop = img[by+y_off : by+y_off+26, bx+x_off : bx+x_off+18]
                gray = cv2.cvtColor(suit_crop, cv2.COLOR_BGR2GRAY)
                
                # Calculate template match for all 4 suits
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
                
                # Get color metrics
                hsv = cv2.cvtColor(suit_crop, cv2.COLOR_BGR2HSV)
                lower_red1 = np.array([0, 50, 50])
                upper_red1 = np.array([25, 255, 255])
                lower_red2 = np.array([170, 50, 50])
                upper_red2 = np.array([180, 255, 255])
                mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
                red_ratio = np.sum(mask > 0) / suit_crop.size
                
                best_suit = max(scores, key=scores.get)
                print(f"  y_off={y_off}, x_off={x_off}: red_ratio={red_ratio:.3f} | Best: {best_suit} ({scores[best_suit]:.3f}) | Scores: Spade={scores.get('spade',0):.3f}, Heart={scores.get('heart',0):.3f}, Diamond={scores.get('diamond',0):.3f}, Club={scores.get('club',0):.3f}")

if __name__ == "__main__":
    main()
