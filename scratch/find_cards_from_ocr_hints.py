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
        "1": "T",  # or could be part of 10
    }
    if text in misreads:
        return misreads[text]
    return None

def find_cards_by_ocr_hints():
    analyzer = BridgeAnalyzer(verbose=False)
    img = cv2.imread("debug_captures/1_ui_full.png")
    if img is None:
        print("Error: debug_captures/1_ui_full.png not found")
        return
        
    h_img, w_img = img.shape[:2]
    print(f"Loaded image: {w_img}x{h_img}")
    
    # Scale up for OCR
    fx = 3.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, (0, 0), fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
    
    # We run OCR on the full scaled image to find hint locations
    # PSM 11 is sparse text finder
    config = "--psm 11"
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
    
    candidates = []
    for row in reader:
        if len(row) <= text_idx:
            continue
        text = row[text_idx].strip()
        if not text:
            continue
        conf = float(row[conf_idx])
        if conf < 15:  # Keep low confidence candidates as they might be single letters
            continue
            
        cleaned = clean_rank_candidate(text)
        if cleaned:
            # Map back to original image scale
            cx = (int(row[left_idx]) + int(row[width_idx]) // 2) / fx
            cy = (int(row[top_idx]) + int(row[height_idx]) // 2) / fx
            w_box = int(row[width_idx]) / fx
            h_box = int(row[height_idx]) / fx
            candidates.append({
                "text": text,
                "cleaned": cleaned,
                "conf": conf,
                "cx": cx,
                "cy": cy,
                "w": w_box,
                "h": h_box
            })
            
    print(f"Found {len(candidates)} rank character candidates.")
    
    # Sort candidates by cy to group/filter
    candidates.sort(key=lambda c: c["cy"])
    
    detected_cards = []
    
    # To avoid duplicate card detections for the same physical card, 
    # we filter out overlapping crops using a minimum distance check (e.g. 15px)
    used_centers = []
    
    for c in candidates:
        cx, cy = c["cx"], c["cy"]
        
        # Check distance to existing detected cards
        duplicate = False
        for ux, uy in used_centers:
            if abs(cx - ux) < 15 and abs(cy - uy) < 15:
                duplicate = True
                break
        if duplicate:
            continue
            
        # Determine Card Region around (cx, cy)
        # Assuming typical trick/dummy card size: 42x66
        # Let's crop a slightly wider region to ensure full coverage, e.g. 44x68
        card_w, card_h = 42, 66
        
        # If the rank character is at the top-center of the card:
        # vertical center of rank is at ~18px from card top (approx 27% of height)
        # horizontal center of rank is at ~21px from card left (approx 50% of width)
        card_x1 = int(cx - 21)
        card_y1 = int(cy - 19)
        card_x2 = card_x1 + card_w
        card_y2 = card_y1 + card_h
        
        # Clip to image boundaries
        card_x1 = max(0, min(card_x1, w_img - 1))
        card_y1 = max(0, min(card_y1, h_img - 1))
        card_x2 = max(0, min(card_x2, w_img))
        card_y2 = max(0, min(card_y2, h_img))
        
        if (card_x2 - card_x1) < 20 or (card_y2 - card_y1) < 30:
            continue
            
        card_crop = img[card_y1:card_y2, card_x1:card_x2]
        
        # Extract rank and suit using the analyzer's centered logic (is_hand=False)
        rank, suit = analyzer.extract_card(card_crop, is_hand=False)
        
        if rank and suit:
            detected_cards.append({
                "rank": rank,
                "suit": suit,
                "cx": cx,
                "cy": cy,
                "bbox": {"x": card_x1, "y": card_y1, "w": card_x2 - card_x1, "h": card_y2 - card_y1}
            })
            used_centers.append((cx, cy))
            
    # Categorize cards into areas using dynamic proportions of the image dimensions:
    # 1. South hand: cy >= 0.75 * h_img
    # 2. North Dummy hand: cy < 0.22 * h_img and 0.20 * w_img <= cx < 0.80 * w_img
    # 3. West Dummy hand: cx < 0.22 * w_img and 0.22 * h_img <= cy < 0.75 * h_img
    # 4. East Dummy hand: cx >= 0.78 * w_img and 0.22 * h_img <= cy < 0.75 * h_img
    # 5. Trick cards: 0.22 * w_img <= cx < 0.78 * w_img and 0.22 * h_img <= cy < 0.75 * h_img
    west_dummy = []
    east_dummy = []
    north_dummy = []
    trick_cards = []
    south_hand = []
    other_cards = []
    
    for card in detected_cards:
        cx, cy = card["cx"], card["cy"]
        if cy >= 0.75 * h_img:
            south_hand.append(card)
        elif cy < 0.22 * h_img and 0.20 * w_img <= cx < 0.80 * w_img:
            north_dummy.append(card)
        elif cx < 0.22 * w_img and 0.22 * h_img <= cy < 0.75 * h_img:
            west_dummy.append(card)
        elif cx >= 0.78 * w_img and 0.22 * h_img <= cy < 0.75 * h_img:
            east_dummy.append(card)
        elif 0.22 * w_img <= cx < 0.78 * w_img and 0.22 * h_img <= cy < 0.75 * h_img:
            trick_cards.append(card)
        else:
            other_cards.append(card)
            
    print("\n--- DETECTED CARDS ---")
    print(f"West Dummy Hand ({len(west_dummy)}):")
    for c in west_dummy:
        print(f"  {c['rank']}{c['suit']} at cx={c['cx']:.1f}, cy={c['cy']:.1f}")
        
    print(f"East Dummy Hand ({len(east_dummy)}):")
    for c in east_dummy:
        print(f"  {c['rank']}{c['suit']} at cx={c['cx']:.1f}, cy={c['cy']:.1f}")
        
    print(f"North Dummy Hand ({len(north_dummy)}):")
    for c in north_dummy:
        print(f"  {c['rank']}{c['suit']} at cx={c['cx']:.1f}, cy={c['cy']:.1f}")
        
    print(f"Trick Cards ({len(trick_cards)}):")
    for c in trick_cards:
        print(f"  {c['rank']}{c['suit']} at cx={c['cx']:.1f}, cy={c['cy']:.1f}")
        
    print(f"South Hand ({len(south_hand)}):")
    for c in south_hand:
        print(f"  {c['rank']}{c['suit']} at cx={c['cx']:.1f}, cy={c['cy']:.1f}")
        
    print(f"Other ({len(other_cards)}):")
    for c in other_cards:
        print(f"  {c['rank']}{c['suit']} at cx={c['cx']:.1f}, cy={c['cy']:.1f}")

if __name__ == "__main__":
    find_cards_by_ocr_hints()
