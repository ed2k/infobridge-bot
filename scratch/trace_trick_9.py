import cv2
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from analyzer import BridgeAnalyzer

def trace_extract_multiple_cards(analyzer, cards_img):
    gray = cv2.cvtColor(cards_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Total contours: {len(contours)}")
    detected_cards = []
    card_idx = 0
    for idx, c in enumerate(contours):
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w)/h
        print(f"Contour {idx}: bbox=({x}, {y}, {w}, {h}), area={area:.1f}, aspect={aspect_ratio:.2f}")
        if area < 1000 or area > 8000:
            continue
        if aspect_ratio < 0.4 or aspect_ratio > 1.2:
            continue
            
        card_crop = cards_img[y:y+h, x:x+w]
        print(f"\n--- Card Contour {card_idx}: bbox=({x}, {y}, {w}, {h}), aspect={aspect_ratio:.2f} ---")
        
        # Trace extract_card logic
        # 1. Split top/bottom
        top_65 = card_crop[0:int(h*0.65), :]
        cg = cv2.cvtColor(top_65, cv2.COLOR_BGR2GRAY)
        _, cb = cv2.threshold(cg, 127, 255, cv2.THRESH_BINARY_INV)
        row_sums = np.sum(cb > 0, axis=1)
        
        search_start = int(h * 0.40)
        search_end = int(h * 0.62)
        best_y = int(h * 0.50)
        min_sum = float('inf')
        for sy in range(search_start, search_end):
            window_sum = np.sum(row_sums[sy-1:sy+2])
            if window_sum < min_sum:
                min_sum = window_sum
                best_y = sy
                
        rank_crop = card_crop[2:best_y, int(w*0.10):int(w*0.90)]
        suit_crop = card_crop[best_y:int(h*0.95), int(w*0.10):int(w*0.90)]
        
        cv2.imwrite(f"debug/trick_card_{card_idx}_full.png", card_crop)
        cv2.imwrite(f"debug/trick_card_{card_idx}_rank.png", rank_crop)
        cv2.imwrite(f"debug/trick_card_{card_idx}_suit.png", suit_crop)
        
        # 2. Run OCR
        import pytesseract
        def normalize_rank_text(raw_text):
            rank_text = raw_text.strip().upper().replace(" ", "")
            if not rank_text:
                return None
            if "10" in rank_text:
                return "T"
            if rank_text == "1":
                return "T"
            if rank_text in ["0", "O", "D"]:
                return "Q"
            valid_ranks = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
            return rank_text if rank_text in valid_ranks else None

        rank_text = None
        for fx_val in [5.0, 4.0, 3.0]:
            processed_rank = analyzer.preprocess_for_ocr(rank_crop, fx=fx_val)
            for psm in [8, 10, 6]:
                custom_config = f"--psm {psm} -c tessedit_char_whitelist=AKQJT1098765432"
                try:
                    raw_rank = pytesseract.image_to_string(processed_rank, config=custom_config)
                    rank_text = normalize_rank_text(raw_rank)
                    if rank_text:
                        print(f"  OCR raw (fx={fx_val}, psm={psm}): '{raw_rank.strip()}' -> '{rank_text}'")
                        break
                except Exception:
                    pass
            if rank_text:
                break

        print(f"  Normalized OCR Rank: {rank_text}")
        
        # Disambiguate 9 vs Q
        if rank_text in ("9", "Q"):
            gray_rank = cv2.cvtColor(rank_crop, cv2.COLOR_BGR2GRAY)
            rh, rw = gray_rank.shape
            bottom = gray_rank[rh//2:, :]
            _, binary = cv2.threshold(bottom, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            left_ink = np.sum(binary[:, :rw//2] > 0)
            right_ink = np.sum(binary[:, rw//2:] > 0)
            bottom_ink = left_ink + right_ink
            ratio = right_ink / max(1, left_ink)
            print(f"  Disambiguation 9/Q ink stats:")
            print(f"    bottom_ink: {bottom_ink}")
            print(f"    left_ink: {left_ink}, right_ink: {right_ink}")
            print(f"    ratio (right/left): {ratio:.3f}")
            
            if bottom_ink > 50:
                if ratio < 1.45:
                    resolved_rank = "Q"
                else:
                    resolved_rank = "9"
            else:
                if rank_text == "9" and right_ink > left_ink * 1.5:
                    resolved_rank = "Q"
                else:
                    resolved_rank = rank_text
            print(f"    resolved to: {resolved_rank} (original: {rank_text})")
            rank_text = resolved_rank

        suit = analyzer.classify_suit_template_matching(suit_crop, is_hand=False)
        print(f"  Detected Rank: {rank_text}, Suit: {suit}")
        card_idx += 1

def main():
    img_path = "debug/3_trick.png"
    img = cv2.imread(img_path)
    print(f"Loaded trick image: {img_path}, shape: {img.shape}")
    analyzer = BridgeAnalyzer(verbose=True)
    trace_extract_multiple_cards(analyzer, img)

if __name__ == "__main__":
    main()
