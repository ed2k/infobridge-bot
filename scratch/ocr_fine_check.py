import cv2
import pytesseract
import numpy as np
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
    analyzer = BridgeAnalyzer(verbose=False)
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
    
    for idx, (bx, by, bw, bh, suit, num_cards) in enumerate(blocks):
        print(f"\n=== Block {idx+1} ({suit.upper()}) ===")
        card_start_x = bx + 25  # skip the suit symbol
        card_area_w = bw - 25 - 55
        
        step = card_area_w / (num_cards - 1) if num_cards > 1 else 0
        print(f"  card_start_x={card_start_x}, card_area_w={card_area_w}, step={step:.1f}")
        
        for i in range(num_cards):
            cx = int(card_start_x + i * step)
            # Crop rank: y = by+2..by+28, x = cx+2..cx+24
            rank_crop = img[by+2 : by+28, cx+2 : cx+24]
            
            detected_rank = None
            for fx_val in [5.0, 4.0, 3.0]:
                gray = cv2.cvtColor(rank_crop, cv2.COLOR_BGR2GRAY)
                scaled = cv2.resize(gray, (0, 0), fx=fx_val, fy=fx_val, interpolation=cv2.INTER_CUBIC)
                thresh = cv2.threshold(scaled, 150, 255, cv2.THRESH_BINARY)[1]
                
                for psm in [10, 8, 6]:
                    config = f"--psm {psm} -c tessedit_char_whitelist=AKQJT1098765432"
                    try:
                        txt = pytesseract.image_to_string(thresh, config=config)
                        r = clean_rank(txt)
                        if r:
                            detected_rank = r
                            break
                    except Exception:
                        pass
                if detected_rank:
                    break
            
            print(f"    Card {i+1} at cx={cx}: Rank={detected_rank or '?'}")

if __name__ == "__main__":
    main()
