import cv2
import numpy as np
import pytesseract
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

def detect_north_dummy_cards(img, analyzer):
    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # We look for candidate dummy suit blocks or suit symbols in y < 250
    # Let's find all contours in y < 250 with area > 100
    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if y < 250 and cv2.contourArea(c) > 100 and w < 300 and h < 110:
            candidates.append((x, y, w, h))
            
    # Sort left-to-right
    candidates.sort()
    
    # Map candidates to suits based on x position:
    # Spade: x < 130
    # Heart: 130 <= x < 260
    # Diamond: 260 <= x < 380
    # Club: 380 <= x
    suits_found = {"spade": None, "heart": None, "diamond": None, "club": None}
    
    for (x, y, w, h) in candidates:
        if x < 130:
            suit = "spade"
        elif x < 260:
            suit = "heart"
        elif x < 380:
            suit = "diamond"
        else:
            suit = "club"
            
        # Keep the largest contour in each suit area as the suit block
        existing = suits_found[suit]
        if existing is None or (w * h) > (existing[2] * existing[3]):
            suits_found[suit] = (x, y, w, h)
            
    detected_cards = []
    
    for suit in ["spade", "heart", "diamond", "club"]:
        block = suits_found[suit]
        if not block:
            continue
            
        bx, by, bw, bh = block
        # If the block height is small or width is small, it's a void suit symbol
        if bh < 50 or bw < 50:
            print(f"  {suit.capitalize()} is void (symbol contour at x={bx}, y={by}, w={bw}, h={bh})")
            continue
            
        # Estimate card count
        card_w = 55
        if bw <= 80:
            num_cards = 1
        else:
            num_cards = int(round((bw - 25 - card_w) / 8.5)) + 1
            
        step = (bw - 25 - card_w) / (num_cards - 1) if num_cards > 1 else 0
        print(f"  {suit.capitalize()}: x={bx}..{bx+bw}, w={bw} -> Estimated {num_cards} cards (step={step:.1f}px)")
        
        card_start_x = bx + 25
        for i in range(num_cards):
            cx = int(card_start_x + i * step)
            rank_crop = img[by+2 : by+28, cx+2 : cx+24]
            
            # Simplified parameter sweep for robust OCR
            candidates = []
            for fx_val in [4.0, 5.0]:
                for thresh_val in [120, 150, 180, "otsu"]:
                    gray_crop = cv2.cvtColor(rank_crop, cv2.COLOR_BGR2GRAY)
                    scaled = cv2.resize(gray_crop, (0, 0), fx=fx_val, fy=fx_val, interpolation=cv2.INTER_CUBIC)
                    
                    if thresh_val == "otsu":
                        thresh_crop = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                    else:
                        thresh_crop = cv2.threshold(scaled, thresh_val, 255, cv2.THRESH_BINARY)[1]
                        
                    for psm in [10, 8, 6]:
                        config = f"--psm {psm} -c tessedit_char_whitelist=AKQJT1098765432"
                        try:
                            txt = pytesseract.image_to_string(thresh_crop, config=config)
                            r = clean_rank(txt)
                            if r:
                                candidates.append(r)
                        except Exception:
                            pass
            
            detected_rank = None
            if candidates:
                from collections import Counter
                detected_rank = Counter(candidates).most_common(1)[0][0]
                    
            detected_cards.append({
                "rank": detected_rank,
                "suit": suit,
                "cx": cx + 11,
                "cy": by + 13
            })
            
    return detected_cards

def main():
    analyzer = BridgeAnalyzer(verbose=False)
    img = cv2.imread("debug_captures/live_ui_all_sides.png")
    if img is None:
        print("Error")
        return
        
    print("Detecting North dummy cards...")
    cards = detect_north_dummy_cards(img, analyzer)
    print(f"\nTotal detected North dummy cards: {len(cards)}")
    for idx, c in enumerate(cards):
        print(f"  Card {idx+1}: {c['rank'] or '?'}{c['suit']} at cx={c['cx']:.1f}, cy={c['cy']:.1f}")

if __name__ == "__main__":
    main()
