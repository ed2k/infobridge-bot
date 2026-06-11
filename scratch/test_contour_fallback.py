import cv2
from collections import Counter
import pytesseract

def clean_rank_candidate(text):
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

def test_fallback():
    img = cv2.imread("debug_captures/1_ui_full.png")
    if img is None:
        print("Failed to load image")
        return
        
    h_img, w_img = img.shape[:2]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if y < 250 and cv2.contourArea(c) > 100 and w < 300 and h < 110:
            candidates.append((x, y, w, h))
            
    candidates.sort()
    
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
            
        existing = suits_found[suit]
        if existing is None or (w * h) > (existing[2] * existing[3]):
            suits_found[suit] = (x, y, w, h)
            
    detected_cards = []
    for suit in ["spade", "heart", "diamond", "club"]:
        block = suits_found[suit]
        if not block:
            continue
            
        bx, by, bw, bh = block
        if bh < 50 or bw < 50:
            continue
            
        card_w = 55
        if bw <= 80:
            num_cards = 1
        else:
            num_cards = int(round((bw - 25 - card_w) / 8.5)) + 1
            
        step = (bw - 25 - card_w) / (num_cards - 1) if num_cards > 1 else 0
        card_start_x = bx + 25
        
        print(f"\nSuit: {suit} at bx={bx}, bw={bw} -> num_cards={num_cards}")
        
        for i in range(num_cards):
            cx = int(card_start_x + i * step)
            rank_crop = img[by+2 : by+28, cx+2 : cx+24]
            
            sweep_candidates = []
            gray_crop = cv2.cvtColor(rank_crop, cv2.COLOR_BGR2GRAY)
            scaled = cv2.resize(gray_crop, (0, 0), fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
            thresh_crop = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            
            try:
                txt = pytesseract.image_to_string(thresh_crop, config="--psm 10 -c tessedit_char_whitelist=AKQJT1098765432")
                r = clean_rank_candidate(txt)
                if r:
                    sweep_candidates.append(r)
            except Exception:
                pass
                
            detected_rank = None
            if sweep_candidates:
                detected_rank = Counter(sweep_candidates).most_common(1)[0][0]
                
            print(f"  Card {i+1} at cx={cx}: Rank={detected_rank or '?'}")

if __name__ == "__main__":
    test_fallback()
