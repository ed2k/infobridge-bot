import cv2
import pytesseract
import csv
from io import StringIO
import os
import numpy as np
from analyzer import BridgeAnalyzer

def clean_rank_candidate(text):
    text = text.strip().upper()
    if not text:
        return None
    # Check if text is a valid rank
    valid_ranks = {"A", "K", "Q", "J", "T", "10", "9", "8", "7", "6", "5", "4", "3", "2"}
    if text in valid_ranks:
        return text
    # Common misreads
    misreads = {
        "0": "Q", "O": "Q", "D": "Q",
        "1": "T",  # or 10
        "J": "J", "L": "J", "I": "J",
    }
    if text in misreads:
        return misreads[text]
    return None

def test_pipeline():
    analyzer = BridgeAnalyzer(verbose=False)
    img = cv2.imread("debug_captures/1_ui_full.png")
    if img is None:
        print("Error: debug_captures/1_ui_full.png not found")
        return
        
    h_img, w_img = img.shape[:2]
    print(f"Loaded image: {w_img}x{h_img}")
    
    fx = 3.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # We will gather candidates from multiple OCR runs with different binarization and PSM modes
    raw_candidates = []
    
    # Run combinations
    # 1. Thresh=200, Invert=True, PSM=11
    # 2. Thresh=220, Invert=True, PSM=6
    # 3. Thresh=180, Invert=True, PSM=6
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
                    raw_candidates.append((cleaned, cx, cy, conf))
        except Exception as e:
            print(f"Error in OCR run ({thresh_val}, {invert}, {psm}): {e}")
            
    print(f"Total raw candidates before merging: {len(raw_candidates)}")
    
    # Merge candidates close to each other
    unique_candidates = []
    for cand in raw_candidates:
        cleaned, cx, cy, conf = cand
        # Check if already added a nearby candidate
        duplicate = False
        for idx, (uc_clean, uc_cx, uc_cy, uc_conf) in enumerate(unique_candidates):
            if abs(cx - uc_cx) < 15 and abs(cy - uc_cy) < 15:
                duplicate = True
                # If the new one has higher confidence, update it
                if conf > uc_conf:
                    unique_candidates[idx] = (cleaned, cx, cy, conf)
                break
        if not duplicate:
            unique_candidates.append((cleaned, cx, cy, conf))
            
    print(f"Unique candidates after merging: {len(unique_candidates)}")
    
    detected_cards = []
    for rank_hint, cx, cy, conf in unique_candidates:
        # Determine Card Crop coordinates
        card_w, card_h = 42, 66
        card_x1 = int(cx - 21)
        card_y1 = int(cy - 19)
        card_x2 = card_x1 + card_w
        card_y2 = card_y1 + card_h
        
        # Clip
        card_x1 = max(0, min(card_x1, w_img - 1))
        card_y1 = max(0, min(card_y1, h_img - 1))
        card_x2 = max(0, min(card_x2, w_img))
        card_y2 = max(0, min(card_y2, h_img))
        
        if (card_x2 - card_x1) < 20 or (card_y2 - card_y1) < 30:
            continue
            
        card_crop = img[card_y1:card_y2, card_x1:card_x2]
        
        # Run card extraction
        rank, suit = analyzer.extract_card(card_crop, is_hand=False)
        
        # If extraction is successful, store the result
        if rank and suit:
            detected_cards.append({
                "rank": rank,
                "suit": suit,
                "cx": cx,
                "cy": cy,
                "bbox": {"x": card_x1, "y": card_y1, "w": card_x2 - card_x1, "h": card_y2 - card_y1}
            })
            
    # Group by region
    # West Dummy: cx < 68, 350 <= cy <= 780
    # Trick area: 68 <= cx < 302, 350 <= cy <= 780
    # South hand: cy >= 650
    west_dummy = []
    trick_cards = []
    south_hand = []
    other_cards = []
    
    for card in detected_cards:
        cx, cy = card["cx"], card["cy"]
        if cy >= 650:
            south_hand.append(card)
        elif cx < 68 and 350 <= cy <= 780:
            west_dummy.append(card)
        elif 68 <= cx < 302 and 350 <= cy <= 780:
            trick_cards.append(card)
        else:
            other_cards.append(card)
            
    print("\n--- DETECTED CARDS BY REGION ---")
    print(f"West Dummy Hand ({len(west_dummy)}):")
    for c in sorted(west_dummy, key=lambda x: (x["cy"], x["cx"])):
        print(f"  {c['rank']}{c['suit']} at cx={c['cx']:.1f}, cy={c['cy']:.1f}")
        
    print(f"Trick Cards ({len(trick_cards)}):")
    for c in sorted(trick_cards, key=lambda x: (x["cy"], x["cx"])):
        print(f"  {c['rank']}{c['suit']} at cx={c['cx']:.1f}, cy={c['cy']:.1f}")
        
    print(f"South Hand ({len(south_hand)}):")
    for c in sorted(south_hand, key=lambda x: (x["cx"], x["cy"])):
        print(f"  {c['rank']}{c['suit']} at cx={c['cx']:.1f}, cy={c['cy']:.1f}")
        
    print(f"Other ({len(other_cards)}):")
    for c in sorted(other_cards, key=lambda x: (x["cy"], x["cx"])):
        print(f"  {c['rank']}{c['suit']} at cx={c['cx']:.1f}, cy={c['cy']:.1f}")

if __name__ == "__main__":
    test_pipeline()
