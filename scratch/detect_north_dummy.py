import cv2
import numpy as np
import os
import pytesseract
from analyzer import BridgeAnalyzer

def clean_rank_text(raw_text):
    text = raw_text.strip().upper().replace(" ", "")
    if not text:
        return None
    if "10" in text:
        return "T"
    if text == "1":
        return "T"
    if text in ["0", "O", "D"]:
        return "Q"
    valid_ranks = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
    return text if text in valid_ranks else None

def main():
    analyzer = BridgeAnalyzer(verbose=False)
    img = cv2.imread("debug_captures/live_ui_all_sides.png")
    if img is None:
        print("Error: debug_captures/live_ui_all_sides.png not found")
        return
        
    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Find contours in the top area (y < 250)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter for candidate dummy suit blocks: y < 250, height around 80-90, width >= 40
    blocks = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if y < 250 and 70 <= h <= 100 and w >= 35:
            blocks.append((x, y, w, h))
            
    # Sort blocks from left to right
    blocks.sort()
    
    print(f"Found {len(blocks)} North dummy suit blocks.")
    
    all_dummy_cards = []
    
    for idx, (bx, by, bw, bh) in enumerate(blocks):
        # Estimate number of cards in this block
        # Dummy card width is typically 55px. Spacing/step is around 18-22px.
        card_w = 55
        if bw <= card_w + 5:
            num_cards = 1
        else:
            num_cards = int(round((bw - card_w) / 20.0)) + 1
            
        step = (bw - card_w) / (num_cards - 1) if num_cards > 1 else 0
        
        print(f"\nBlock {idx+1}: x={bx}..{bx+bw} (w={bw}), y={by}..{by+bh} (h={bh}) -> Estimated {num_cards} cards (step={step:.1f}px)")
        
        # Determine the suit of the block by looking at the first card's suit area
        # The suit symbol is located in the top-left region of the first card, e.g. y = by + 28..58, x = bx + 5..25
        suit_crop = img[by+28:by+58, bx+5:bx+25]
        suit = analyzer.classify_suit_template_matching(suit_crop)
        if not suit:
            suit = analyzer.classify_suit_by_color_shape(suit_crop)
            
        print(f"  Classified block suit: {suit}")
        
        # Slices cards
        for i in range(num_cards):
            cx_card = int(bx + i * step)
            # Crop the rank of this card (top-left portion of the card)
            # Rank is typically y = by + 2..28, x = cx_card + 3..25
            rank_crop = img[by+2:by+28, cx_card+3:cx_card+25]
            
            # Run OCR on rank
            rank = None
            for fx_val in [5.0, 4.0, 3.0]:
                processed_rank = analyzer.preprocess_for_ocr(rank_crop, fx=fx_val)
                for psm in [8, 10, 6]:
                    custom_config = f"--psm {psm} -c tessedit_char_whitelist=AKQJT1098765432"
                    try:
                        raw_rank = pytesseract.image_to_string(processed_rank, config=custom_config)
                        rank = clean_rank_text(raw_rank)
                        if rank:
                            break
                    except Exception:
                        pass
                if rank:
                    break
                    
            print(f"    Card {i+1} at x={cx_card} -> Rank={rank or '?'}")
            all_dummy_cards.append({
                "rank": rank,
                "suit": suit,
                "x": cx_card,
                "y": by
            })
            
    print("\n--- DETECTED NORTH DUMMY CARDS ---")
    suits_dict = {"spade": [], "heart": [], "diamond": [], "club": []}
    for c in all_dummy_cards:
        r = c["rank"] or "?"
        s = c["suit"] or "unknown"
        if s in suits_dict:
            suits_dict[s].append(r)
            
    for s in ["spade", "heart", "diamond", "club"]:
        print(f"  {s.capitalize()}: {', '.join(suits_dict[s])}")

if __name__ == "__main__":
    main()
