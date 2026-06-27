import cv2
import sys
import os
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from analyzer import BridgeAnalyzer

def trace_extract_bids_paddle(analyzer, bidding_img, fx=4.0):
    processed = analyzer.preprocess_for_ocr(bidding_img, fx=fx, thresh_val=None)
    processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    from paddle_ocr import ocr_with_positions
    detections = ocr_with_positions(processed_bgr, min_confidence=0.3)
    if not detections:
        print("No detections at min_confidence=0.3")
        return []

    img_h = bidding_img.shape[0]
    print(f"img_h: {img_h}")

    split_detections = []
    for text, cx, cy, w, h in detections:
        text_clean = text.strip().upper()
        if not text_clean:
            continue
        parts = [p for p in text_clean.split() if p.strip()]
        if len(parts) > 1:
            n_parts = len(parts)
            w_part = w / n_parts
            x_start = cx - w / 2
            for i, part in enumerate(parts):
                part_cx = x_start + w_part * (i + 0.5)
                split_detections.append((part, part_cx, cy, w_part, h))
        else:
            split_detections.append((text_clean, cx, cy, w, h))

    header_candidates = []
    bid_candidates = []

    for text_clean, cx, cy, w, h in split_detections:
        dir_key = analyzer.clean_header_text(text_clean)
        if dir_key and (cy / fx) < max(80.0, img_h * 0.35):
            header_candidates.append((cx / fx, cy / fx, dir_key))
        bid_candidates.append((cx / fx, cy / fx, text_clean, w / fx, h / fx))

    print(f"Header candidates: {header_candidates}")
    if len(header_candidates) < 2:
        print("Fewer than 2 header candidates found.")
        return []

    header_candidates.sort(key=lambda x: x[0])
    min_cx = header_candidates[0][0]
    spacings = [header_candidates[i+1][0] - header_candidates[i][0]
                 for i in range(len(header_candidates) - 1)]
    col_width = sum(spacings) / len(spacings) if spacings else 150.0

    full_order = ["N", "E", "S", "W"]
    best_rotation = full_order
    best_c0 = min_cx
    max_matches = -1

    for leftmost_slot in range(4):
        candidate_c0 = min_cx - (leftmost_slot * col_width)
        for r in range(4):
            candidate_rotation = full_order[r:] + full_order[:r]
            matches = 0
            for cx, cy, direction in header_candidates:
                slot_idx = int(round((cx - candidate_c0) / col_width))
                if 0 <= slot_idx < 4 and candidate_rotation[slot_idx] == direction:
                    matches += 1
            if matches > max_matches:
                max_matches = matches
                best_rotation = candidate_rotation
                best_c0 = candidate_c0

    col_centers = [best_c0 + idx * col_width for idx in range(4)]
    col_dirs = best_rotation
    max_header_top = max(cy for cx, cy, direction in header_candidates)

    print(f"Reconstructed Columns centers: {col_centers}, dirs: {col_dirs}, max_header_top: {max_header_top}")

    candidate_words = []
    for cx, cy, text, w, h in bid_candidates:
        if cy <= max_header_top + 2.5:
            print(f"Skipping header word: {text} at cy={cy:.1f} <= max_header_top+2.5={max_header_top+2.5:.1f}")
            continue

        closest_idx = 0
        min_dist = float('inf')
        for idx, center in enumerate(col_centers):
            dist = abs(cx - center)
            if dist < min_dist:
                min_dist = dist
                closest_idx = idx

        std_text = analyzer.standardize_bid(text)
        print(f"Candidate: raw='{text}', std='{std_text}', cy={cy:.1f}, cx={cx:.1f}")
        
        # Resolve suit symbol images in level bids (e.g. 1H, 1S)
        if len(std_text) >= 1 and std_text[0] in "1234567" and std_text[1:] != "NT":
            has_valid_ocr_suit = (len(std_text) == 2 and std_text[1] in ("S", "H", "D", "C"))
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            x1 = max(0, min(x1, bidding_img.shape[1] - 1))
            y1 = max(0, min(y1, bidding_img.shape[0] - 1))
            x2 = max(0, min(x2, bidding_img.shape[1]))
            y2 = max(0, min(y2, bidding_img.shape[0]))
            
            if x2 > x1 and y2 > y1:
                word_crop = bidding_img[y1:y2, x1:x2]
                suit_w = int(word_crop.shape[1] * 0.6)
                if suit_w > 0:
                    suit_crop = word_crop[:, word_crop.shape[1] - suit_w:]
                    suit, score = analyzer.classify_suit_template_matching(suit_crop, return_score=True)
                    
                    should_override = False
                    resolved_suit = None
                    if has_valid_ocr_suit:
                        if suit and score > 0.60:
                            should_override = True
                            resolved_suit = suit
                    else:
                        if suit:
                            should_override = True
                            resolved_suit = suit
                        else:
                            fallback_suit = analyzer.classify_suit_by_color_shape(suit_crop)
                            if fallback_suit in ("spade", "heart", "diamond", "club"):
                                should_override = True
                                resolved_suit = fallback_suit
                    
                    if should_override and resolved_suit in ("spade", "heart", "diamond", "club"):
                        suit_map = {"spade": "S", "heart": "H", "diamond": "D", "club": "C"}
                        std_text = f"{std_text[0]}{suit_map[resolved_suit]}"
                        print(f"  -> Visual override applied: std='{std_text}'")

        bid_pattern = re.compile(
            r'^(PASS|PAS|PA|PASSED|DBL|DOUBLE|RDBL|REDOUBLE|X|XX|'
            r'[1-7]\s*(?:NT|N|S|H|D|C|SPADES|HEARTS|DIAMONDS|CLUBS))$',
            re.IGNORECASE
        )

        if bid_pattern.match(std_text):
            direction = col_dirs[closest_idx]
            bbox = {"x": cx - w/2, "y": cy - h/2, "w": w, "h": h}
            candidate_words.append({
                "cy": cy,
                "cx": cx,
                "col_idx": closest_idx,
                "direction": direction,
                "text": std_text,
                "bbox": bbox
            })
        else:
            print(f"  -> Skipped: raw='{text}' std='{std_text}' does not match bid_pattern")

    candidate_words.sort(key=lambda w: w["cy"])
    print("\n--- Rows Grouping ---")
    word_rows = []
    if candidate_words:
        current_row = [candidate_words[0]]
        for w in candidate_words[1:]:
            if abs(w["cy"] - current_row[-1]["cy"]) < (24.0 / fx):
                current_row.append(w)
            else:
                word_rows.append(current_row)
                current_row = [w]
        word_rows.append(current_row)

    for i, r in enumerate(word_rows):
        print(f"Row {i}: {[w['text'] for w in r]} at average cy={sum(w['cy'] for w in r)/len(r):.1f}")

    print("\n--- Input Area Filtering ---")
    input_area_y = None
    for i, row in enumerate(word_rows):
        is_btn_row = False
        single_digits = [w for w in row if w["text"].strip() in "1234567"]
        if len(single_digits) >= 3:
            is_btn_row = True
            reason = "3+ single digits"
        else:
            reason = "other check"
            for w in row:
                cleaned = re.sub(r'\s+', '', w["text"])
                if len(cleaned) >= 4 and all(c in "1234567" for c in cleaned):
                    is_btn_row = True
                    reason = "cleaned >= 4 and all in 1234567"
                    break
                if any(seq in cleaned for seq in ["1234567", "12345", "23456", "34567"]):
                    is_btn_row = True
                    reason = "seq match"
                    break
                if cleaned in "1234567" and w["cy"] > bidding_img.shape[0] * 0.55:
                    is_btn_row = True
                    reason = "cleaned in 1234567 and cy > 0.55"
                    break
        print(f"Row {i}: is_btn_row={is_btn_row} (Reason: {reason})")
        if is_btn_row:
            row_min_cy = min(w["cy"] for w in row)
            if input_area_y is None or row_min_cy < input_area_y:
                input_area_y = row_min_cy
    print(f"Final input_area_y: {input_area_y}")

    valid_rows = []
    for i, row in enumerate(word_rows):
        row_cy = sum(w["cy"] for w in row) / len(row)
        if input_area_y is not None and row_cy >= input_area_y - 2:
            print(f"Row {i} skipped: row_cy={row_cy:.1f} >= input_area_y={input_area_y:.1f}")
            continue
        if row_cy > bidding_img.shape[0] * 0.80:
            print(f"Row {i} skipped: row_cy={row_cy:.1f} > bottom_threshold={bidding_img.shape[0] * 0.80:.1f}")
            continue
        valid_rows.append(row)

    print("\n--- Final Results ---")
    results = []
    for row in valid_rows:
        col_groups = {}
        for w in row:
            col_groups.setdefault(w["col_idx"], []).append(w)
        for col_idx in sorted(col_groups.keys()):
            g_words = col_groups[col_idx]
            g_words.sort(key=lambda w: w["cx"])
            combined_text = " ".join(w["text"] for w in g_words)
            std_text = analyzer.standardize_bid(combined_text)
            first_w = g_words[0]
            direction = first_w["direction"]
            results.append((direction, std_text))
    print(f"Results: {results}")

def main():
    img_path = "debug/bidding_area.png"
    img = cv2.imread(img_path)
    analyzer = BridgeAnalyzer(verbose=True)
    trace_extract_bids_paddle(analyzer, img)

if __name__ == "__main__":
    main()
