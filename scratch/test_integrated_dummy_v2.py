import cv2
import numpy as np
import pytesseract

def detect_dummy_hands_v2(img):
    if img is None:
        return {"West": [], "North": [], "East": []}
        
    h_img, w_img = img.shape[:2]
    
    best_total = -1
    detected_cards = []
    
    if h_img >= 310:
        crop_w = w_img
        dummy_text_crop = img[275:310, 0:crop_w]
        
        # Determine suit order dynamically by checking colors in Column 3 vs Column 4
        col3_x1 = int(220 * (crop_w / 510.0))
        col3_x2 = int(330 * (crop_w / 510.0))
        col4_x1 = int(330 * (crop_w / 510.0))
        col4_x2 = int(480 * (crop_w / 510.0))
        
        crop_col3 = dummy_text_crop[:, col3_x1:col3_x2]
        crop_col4 = dummy_text_crop[:, col4_x1:col4_x2]
        
        b3, g3, r3 = cv2.split(crop_col3)
        red3 = np.sum((r3.astype(int) > g3.astype(int) + 40) & (r3.astype(int) > b3.astype(int) + 40))
        
        b4, g4, r4 = cv2.split(crop_col4)
        red4 = np.sum((r4.astype(int) > g4.astype(int) + 40) & (r4.astype(int) > b4.astype(int) + 40))
        
        if red3 >= red4:
            suits_order = ["spade", "heart", "diamond", "club"]
        else:
            suits_order = ["spade", "heart", "club", "diamond"]
            
        print(f"Detected Suits Order: {suits_order}")
            
        # Define column boundaries
        cols = [
            (0, int(130 * (crop_w / 510.0))),
            (int(130 * (crop_w / 510.0)), int(250 * (crop_w / 510.0))),
            (int(250 * (crop_w / 510.0)), int(370 * (crop_w / 510.0))),
            (int(370 * (crop_w / 510.0)), crop_w)
        ]
        
        configs = [
            (4.0, "otsu", True, 6),
            (3.0, "otsu", True, 6),
        ]
        
        for fx_val, thresh_val, invert_val, psm_val in configs:
            current_detected = []
            for col_idx, (x1, x2) in enumerate(cols):
                col_crop = dummy_text_crop[:, x1:x2]
                col_gray = cv2.cvtColor(col_crop, cv2.COLOR_BGR2GRAY)
                
                scaled = cv2.resize(col_gray, (0, 0), fx=fx_val, fy=fx_val, interpolation=cv2.INTER_CUBIC)
                if thresh_val == "otsu":
                    thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                else:
                    thresh = cv2.threshold(scaled, thresh_val, 255, cv2.THRESH_BINARY)[1]
                    
                proc = cv2.bitwise_not(thresh) if invert_val else thresh
                
                try:
                    txt = pytesseract.image_to_string(proc, config=f"--psm {psm_val}")
                    
                    mapping = {
                        "0": "Q", "O": "Q", "D": "Q",
                        "S": "J", "N": "T", "W": "T",
                        "Z": "", "E": "6",
                        "M": "", "B": "", "I": "", "F": "", "H": "", "X": "",
                    }
                    cleaned = []
                    text_upper = txt.strip().upper().replace("\n", "").replace(" ", "").replace("10", "T")
                    for char in text_upper:
                        if char in mapping:
                            repl = mapping[char]
                            if repl:
                                cleaned.append(repl)
                        elif char.isdigit() or char in ["A", "K", "Q", "J", "T"]:
                            cleaned.append(char)
                            
                    filtered = []
                    for c in cleaned:
                        if not filtered or filtered[-1] != c:
                            filtered.append(c)
                            
                    rank_order = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
                    suit_cards = [r for r in filtered if r in rank_order]
                    
                    suit_name = suits_order[col_idx]
                    for r in suit_cards:
                        current_detected.append({
                            "rank": r,
                            "suit": suit_name,
                            "cx": 50 + col_idx * 100,
                            "cy": 290,
                            "bbox": {"x": 50 + col_idx * 100, "y": 275, "w": 40, "h": 30}
                        })
                except Exception:
                    pass
            
            total_cards = len(current_detected)
            if 0 < total_cards <= 13:
                if total_cards > best_total:
                    best_total = total_cards
                    detected_cards = current_detected
                    if total_cards == 13:
                        break
                        
        if best_total > 0:
            north_dummy = detected_cards
            return {"West": [], "North": north_dummy, "East": []}
            
    return {"West": [], "North": [], "East": []}

def test_all():
    for img_path in ["debug_captures/live_ui_all_sides.png", "debug_captures/1_ui_full.png"]:
        print(f"\n==================== Testing {img_path} ====================")
        img = cv2.imread(img_path)
        res = detect_dummy_hands_v2(img)
        north = res["North"]
        print(f"North Dummy Hand: {len(north)} cards")
        
        # Group by suit for display
        suits = {"spade": [], "heart": [], "diamond": [], "club": []}
        for c in north:
            suits[c["suit"]].append(c["rank"])
            
        print(f"  Spades:   {suits['spade']}")
        print(f"  Hearts:   {suits['heart']}")
        print(f"  Clubs:    {suits['club']}")
        print(f"  Diamonds: {suits['diamond']}")

if __name__ == "__main__":
    test_all()
