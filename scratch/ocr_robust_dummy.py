import cv2
import pytesseract
import numpy as np
from collections import Counter
from analyzer import BridgeAnalyzer

def clean_rank(text):
    t = text.strip().upper().replace(" ", "")
    if not t:
        return None
    if t == "10":
        return "T"
    if t == "1":
        return "T"
    if t in ["0", "O", "D"]:
        return "Q"
    valid = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
    return t if t in valid else None

def main():
    img = cv2.imread("debug_captures/live_ui_all_sides.png")
    if img is None:
        print("Error")
        return
        
    blocks = [
        # (bx, by, bw, bh, suit, num_cards)
        (19, 75, 114, 85, "spade", 5),
        (142, 75, 114, 85, "heart", 5),
        (279, 75, 100, 85, "diamond", 3)
    ]
    
    # We will save the best detected ranks
    all_dummy = []
    
    for idx, (bx, by, bw, bh, suit, num_cards) in enumerate(blocks):
        print(f"\n=== Block {idx+1} ({suit.upper()}) ===")
        card_start_x = bx + 25  # skip the suit symbol
        card_area_w = bw - 25 - 55
        
        step = card_area_w / (num_cards - 1) if num_cards > 1 else 0
        
        for i in range(num_cards):
            cx = int(card_start_x + i * step)
            # Crop rank: y = by+2..by+28, x = cx+2..cx+24
            rank_crop = img[by+2 : by+28, cx+2 : cx+24]
            
            # Sweep parameters
            candidates = []
            for fx_val in [3.0, 4.0, 5.0, 6.0]:
                for thresh_val in [100, 120, 140, 160, 180, 200, 220, "otsu"]:
                    gray = cv2.cvtColor(rank_crop, cv2.COLOR_BGR2GRAY)
                    scaled = cv2.resize(gray, (0, 0), fx=fx_val, fy=fx_val, interpolation=cv2.INTER_CUBIC)
                    
                    if thresh_val == "otsu":
                        thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                    else:
                        thresh = cv2.threshold(scaled, thresh_val, 255, cv2.THRESH_BINARY)[1]
                        
                    for psm in [6, 8, 10, 13]:
                        config = f"--psm {psm} -c tessedit_char_whitelist=AKQJT1098765432"
                        try:
                            txt = pytesseract.image_to_string(thresh, config=config)
                            r = clean_rank(txt)
                            if r:
                                candidates.append(r)
                        except Exception:
                            pass
            
            # Find the most common rank
            if candidates:
                counter = Counter(candidates)
                best_rank, count = counter.most_common(1)[0]
                print(f"    Card {i+1} at cx={cx}: Best Rank={best_rank} (confidence counts={count}/{len(candidates)}) | candidates={counter}")
                all_dummy.append((suit, best_rank, cx, by))
            else:
                print(f"    Card {i+1} at cx={cx}: No rank detected.")
                all_dummy.append((suit, "?", cx, by))

if __name__ == "__main__":
    main()
