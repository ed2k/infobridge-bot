import cv2
import pytesseract
import csv
from io import StringIO
import os
import numpy as np
from analyzer import BridgeAnalyzer
from capture import ScreenCapture

def clean_rank_candidate(text):
    text = text.strip().upper()
    if not text:
        return None
    valid_ranks = {"A", "K", "Q", "J", "T", "10", "9", "8", "7", "6", "5", "4", "3", "2"}
    if text in valid_ranks:
        return text
    misreads = {
        "0": "Q", "O": "Q", "D": "Q",
        "1": "T",
        "J": "J", "L": "J", "I": "J",
    }
    if text in misreads:
        return misreads[text]
    return None

def main():
    analyzer = BridgeAnalyzer(verbose=False)
    
    img = None
    import sys
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        print(f"Loading image from file: {img_path}")
        img = cv2.imread(img_path)
    else:
        try:
            cap = ScreenCapture()
            img = cap.capture_ui()
        except Exception as e:
            print(f"Capture failed: {e}")
            
    if img is None:
        fallback = "debug_captures/live_ui_all_sides.png"
        if os.path.exists(fallback):
            print(f"Falling back to local image: {fallback}")
            img = cv2.imread(fallback)
            
    if img is None:
        print("Error: Could not capture live UI or load fallback")
        return
        
    h_img, w_img = img.shape[:2]
    print(f"Captured live UI: {w_img}x{h_img}")
    
    fx = 3.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    raw_candidates = []
    
    runs = [
        (200, True, 11),
        (220, True, 6),
        (180, True, 6)
    ]
    
    for thresh_val, invert, psm in runs:
        _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        proc = cv2.bitwise_not(thresh) if invert else thresh
        scaled = cv2.resize(proc, (0, 0), fx=fx, fy=fx, interpolation=cv2.INTER_NEAREST)
        
        config = f"--psm {psm}"
        try:
            data_str = pytesseract.image_to_data(scaled, config=config, output_type=pytesseract.Output.STRING)
            f = StringIO(data_str)
            reader = csv.reader(f, delimiter='\t')
            header = next(reader)
            
            left_idx = header.index('left')
            top_idx = header.index('top')
            width_idx = header.index('width')
            height_idx = header.index('height')
            text_idx = header.index('text')
            conf_idx = header.index('conf')
            
            for row in reader:
                if len(row) <= text_idx:
                    continue
                text = row[text_idx].strip()
                if not text:
                    continue
                conf = float(row[conf_idx])
                if conf < 10:
                    continue
                cleaned = clean_rank_candidate(text)
                if cleaned:
                    cx = (int(row[left_idx]) + int(row[width_idx]) // 2) / fx
                    cy = (int(row[top_idx]) + int(row[height_idx]) // 2) / fx
                    if cy < 0.22 * h_img:
                        continue
                    raw_candidates.append((cleaned, cx, cy, conf))
        except Exception as e:
            print(f"Error in OCR run ({thresh_val}, {invert}, {psm}): {e}")
            
    unique_candidates = []
    for cand in raw_candidates:
        cleaned, cx, cy, conf = cand
        duplicate = False
        for idx, (uc_clean, uc_cx, uc_cy, uc_conf) in enumerate(unique_candidates):
            if abs(cx - uc_cx) < 15 and abs(cy - uc_cy) < 15:
                duplicate = True
                if conf > uc_conf:
                    unique_candidates[idx] = (cleaned, cx, cy, conf)
                break
        if not duplicate:
            unique_candidates.append((cleaned, cx, cy, conf))
            
    detected_cards = []
    for rank_hint, cx, cy, conf in unique_candidates:
        card_w, card_h = 42, 66
        card_x1 = int(cx - 21)
        card_y1 = int(cy - 33)
        card_x2 = card_x1 + card_w
        card_y2 = card_y1 + card_h
        
        card_x1 = max(0, min(card_x1, w_img - 1))
        card_y1 = max(0, min(card_y1, h_img - 1))
        card_x2 = max(0, min(card_x2, w_img))
        card_y2 = max(0, min(card_y2, h_img))
        
        if (card_x2 - card_x1) < 20 or (card_y2 - card_y1) < 30:
            continue
            
        card_crop = img[card_y1:card_y2, card_x1:card_x2]
        rank, suit = analyzer.extract_card(card_crop, is_hand=False)
        
        if not rank and suit:
            rank = rank_hint
            
        if rank and suit:
            detected_cards.append({
                "rank": rank,
                "suit": suit,
                "cx": cx,
                "cy": cy,
                "bbox": {"x": card_x1, "y": card_y1, "w": card_x2 - card_x1, "h": card_y2 - card_y1}
            })
            
    # ----------------------------------------------------
    # Detect North Dummy Hand using Contour Block Slicing
    # ----------------------------------------------------
    contour_dummy_cards = []
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
        
        for i in range(num_cards):
            cx = int(card_start_x + i * step)
            rank_crop = img[by+2 : by+28, cx+2 : cx+24]
            
            # Sweep parameters for robust OCR
            sweep_candidates = []
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
                            r = clean_rank_candidate(txt)
                            if r:
                                sweep_candidates.append(r)
                        except Exception:
                            pass
            
            detected_rank = None
            if sweep_candidates:
                from collections import Counter
                detected_rank = Counter(sweep_candidates).most_common(1)[0][0]
                
            contour_dummy_cards.append({
                "rank": detected_rank,
                "suit": suit,
                "cx": cx + 11,
                "cy": by + 13,
                "bbox": {"x": cx, "y": by, "w": 55, "h": 85}
            })
            
    # ----------------------------------------------------
    # Detect Dummy Hand from compact Text Line (y=275..310)
    # ----------------------------------------------------
    best_suits = []
    best_text = ""
    
    crop_w = w_img
    dummy_text_crop = img[275:310, 0:crop_w]
    gray_dt = cv2.cvtColor(dummy_text_crop, cv2.COLOR_BGR2GRAY)
    
    # Sweep parameters to find a clean 13-card hand representation
    configs = [
        # (fx, thresh_type, invert, psm)
        (3.0, "otsu", True, 6),
        (3.0, "otsu", True, 7),
        (5.0, "otsu", True, 6),
        (5.0, "otsu", True, 7),
        (4.0, "otsu", True, 6),
        (4.0, "otsu", True, 7),
        (3.0, 150, True, 6),
        (3.0, 127, False, 6),
    ]
    
    for fx_val, thresh_val, invert_val, psm_val in configs:
        scaled_dt = cv2.resize(gray_dt, (0, 0), fx=fx_val, fy=fx_val, interpolation=cv2.INTER_CUBIC)
        if thresh_val == "otsu":
            thresh_dt = cv2.threshold(scaled_dt, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        else:
            thresh_dt = cv2.threshold(scaled_dt, thresh_val, 255, cv2.THRESH_BINARY)[1]
            
        proc_dt = cv2.bitwise_not(thresh_dt) if invert_val else thresh_dt
        
        try:
            txt = pytesseract.image_to_string(proc_dt, config=f"--psm {psm_val}")
            txt_clean = txt.strip().replace("\n", " ")
            if not txt_clean:
                continue
                
            mapping = {
                "0": "Q", "O": "Q", "D": "Q",
                "1": "T", "S": "",
                "N": "T", "W": "T",
                "Z": "",
                "E": "6",
                "M": "", "B": "", "I": "", "F": "", "H": "", "X": "",
            }
            cleaned_ranks = []
            text_upper = txt_clean.upper().replace("10", "T")
            for char in text_upper:
                if char.isdigit():
                    cleaned_ranks.append(char)
                elif char in mapping:
                    repl = mapping[char]
                    if repl:
                        cleaned_ranks.append(repl)
                elif char in ["A", "K", "Q", "J", "T"]:
                    cleaned_ranks.append(char)
                    
            # Remove adjacent duplicates
            filtered_ranks = []
            for c in cleaned_ranks:
                if not filtered_ranks or filtered_ranks[-1] != c:
                    filtered_ranks.append(c)
                    
            # Split into suits using descending order rule
            rank_order = {r: idx for idx, r in enumerate(["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"])}
            suits = []
            current_suit = []
            for r in filtered_ranks:
                if r not in rank_order:
                    continue
                if not current_suit:
                    current_suit.append(r)
                else:
                    prev_r = current_suit[-1]
                    if rank_order[r] <= rank_order[prev_r]:
                        suits.append(current_suit)
                        current_suit = [r]
                    else:
                        current_suit.append(r)
            if current_suit:
                suits.append(current_suit)
                
            if len(suits) <= 4:
                total_cards = sum(len(s) for s in suits)
                if total_cards == 13 and len(suits) == 4:
                    best_suits = suits
                    best_text = txt_clean
                    break
                elif total_cards < 13:
                    if not best_suits or total_cards > sum(len(s) for s in best_suits):
                        best_suits = suits
                        best_text = txt_clean
        except Exception:
            pass
            
    if best_suits:
        print(f"Detected compact dummy hand text: '{best_text}'")
        suits_order = ["spade", "heart", "diamond", "club"]
        for idx, suit_cards in enumerate(best_suits):
            if idx >= 4:
                break
            suit_name = suits_order[idx]
            for r in suit_cards:
                detected_cards.append({
                    "rank": r,
                    "suit": suit_name,
                    "cx": 50 + idx * 100,
                    "cy": 290,
                    "bbox": {"x": 50 + idx * 100, "y": 275, "w": 40, "h": 30}
                })
    else:
        # Fall back to contour block slicing detections
        print("Compact dummy hand text not detected, falling back to contour cards.")
        detected_cards.extend(contour_dummy_cards)
            
    west_dummy = []
    east_dummy = []
    north_dummy = []
    trick_cards = []
    south_hand = []
    other_cards = []
    
    for card in detected_cards:
        cx, cy = card["cx"], card["cy"]
        if card.get("bbox", {}).get("y") == 275:
            north_dummy.append(card)
        elif cy >= 0.75 * h_img:
            south_hand.append(card)
        elif cy < 0.22 * h_img:
            north_dummy.append(card)
        elif cx < 0.22 * w_img and 0.22 * h_img <= cy < 0.75 * h_img:
            west_dummy.append(card)
        elif cx >= 0.78 * w_img and 0.22 * h_img <= cy < 0.75 * h_img:
            east_dummy.append(card)
        elif 0.22 * w_img <= cx < 0.78 * w_img and 0.22 * h_img <= cy < 0.75 * h_img:
            trick_cards.append(card)
        else:
            other_cards.append(card)
            
    print("\n--- LIVE DETECTED CARDS BY REGION ---")
    print(f"West Dummy Hand ({len(west_dummy)}):")
    for c in sorted(west_dummy, key=lambda x: (x["cy"], x["cx"])):
        print(f"  {c['rank'] or '?'}{c['suit']} at cx={c['cx']:.1f}, cy={c['cy']:.1f}")
        
    print(f"East Dummy Hand ({len(east_dummy)}):")
    for c in sorted(east_dummy, key=lambda x: (x["cy"], x["cx"])):
        print(f"  {c['rank'] or '?'}{c['suit']} at cx={c['cx']:.1f}, cy={c['cy']:.1f}")
        
    print(f"North Dummy Hand ({len(north_dummy)}):")
    for c in sorted(north_dummy, key=lambda x: (x["cx"], x["cy"])):
        print(f"  {c['rank'] or '?'}{c['suit']} at cx={c['cx']:.1f}, cy={c['cy']:.1f}")
        
    print(f"Trick Cards ({len(trick_cards)}):")
    for c in sorted(trick_cards, key=lambda x: (x["cy"], x["cx"])):
        print(f"  {c['rank'] or '?'}{c['suit']} at cx={c['cx']:.1f}, cy={c['cy']:.1f}")
        
    print(f"South Hand ({len(south_hand)}):")
    for c in sorted(south_hand, key=lambda x: (x["cx"], x["cy"])):
        print(f"  {c['rank'] or '?'}{c['suit']} at cx={c['cx']:.1f}, cy={c['cy']:.1f}")
        
    print(f"Other ({len(other_cards)}):")
    for c in sorted(other_cards, key=lambda x: (x["cy"], x["cx"])):
        print(f"  {c['rank'] or '?'}{c['suit']} at cx={c['cx']:.1f}, cy={c['cy']:.1f}")

if __name__ == "__main__":
    main()
