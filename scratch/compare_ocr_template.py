import cv2
import numpy as np
import os
import pytesseract
from io import StringIO
import csv
from analyzer import BridgeAnalyzer

def nms(boxes, threshold=0.5):
    """Non-Maximum Suppression to filter overlapping detections."""
    if len(boxes) == 0:
        return []
    
    # boxes format: (x, y, w, h, score, label)
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    pick = []
    
    while len(boxes) > 0:
        b = boxes[0]
        pick.append(b)
        boxes = boxes[1:]
        
        remaining = []
        for box in boxes:
            # Calculate IoU
            x1 = max(b[0], box[0])
            y1 = max(b[1], box[1])
            x2 = min(b[0] + b[2], box[0] + box[2])
            y2 = min(b[1] + b[3], box[1] + box[3])
            
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            intersection = w * h
            
            area_b = b[2] * b[3]
            area_box = box[2] * box[3]
            union = area_b + area_box - intersection
            
            iou = intersection / union if union > 0 else 0
            if iou < threshold:
                remaining.append(box)
        boxes = remaining
        
    return pick

def clean_rank_candidate(text):
    text = text.strip().upper().replace(" ", "")
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
    east_path = "debug/dummy_strip_east.png"
    if not os.path.exists(east_path):
        print(f"❌ {east_path} not found.")
        return

    # Create destination dir for artifacts if it doesn't exist
    artifact_dir = "/Users/admin/.gemini/antigravity/brain/6dc85fcf-c4f3-4cef-bc11-8fb451739bbd"
    os.makedirs(artifact_dir, exist_ok=True)

    img = cv2.imread(east_path)
    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    analyzer = BridgeAnalyzer(verbose=False)

    print("=========================================")
    print("METHOD 1: OCR-Based Card Detection (like main.py)")
    print("=========================================")
    
    # 1. Run regional OCR on side column to find rank candidates
    raw_candidates = []
    gray_crop = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Same thresholding and resizing as main.py
    for thresh_val, invert, psm in [(200, True, 11)]:
        _, thresh = cv2.threshold(gray_crop, thresh_val, 255, cv2.THRESH_BINARY)
        proc = cv2.bitwise_not(thresh) if invert else thresh
        scaled = cv2.resize(proc, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_NEAREST)
        
        try:
            data_str = pytesseract.image_to_data(scaled, config=f"--psm {psm}", output_type=pytesseract.Output.STRING)
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
                    cx = (int(row[left_idx]) + int(row[width_idx]) // 2) / 3.0
                    cy = (int(row[top_idx]) + int(row[height_idx]) // 2) / 3.0
                    raw_candidates.append((cleaned, cx, cy, conf))
        except Exception as e:
            print(f"OCR candidate extraction error: {e}")

    # Remove duplicates
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
            
    ocr_detected_cards = []
    print(f"Found {len(unique_candidates)} rank hints via initial OCR.")
    
    # Create copy of image to draw OCR annotations
    img_ocr = img.copy()

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
        
        print(f"  Hint '{rank_hint}' at ({cx:.1f}, {cy:.1f}) -> Crop bbox ({card_x1}, {card_y1}, {card_x2-card_x1}x{card_y2-card_y1})")
        print(f"    Extracted: Rank={rank or 'None'}, Suit={suit or 'None'}")
        
        # Save crop for inspection
        cv2.imwrite(f"debug/ocr_crop_cand_{rank_hint}_{cx:.0f}_{cy:.0f}.png", card_crop)

        # Draw bbox
        cv2.rectangle(img_ocr, (card_x1, card_y1), (card_x2, card_y2), (0, 0, 255), 1)
        # Draw label
        label = f"{rank or '?'}{suit[0].upper() if suit else '?'}"
        cv2.putText(img_ocr, label, (card_x1, card_y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        if rank and suit:
            ocr_detected_cards.append({
                "rank": rank,
                "suit": suit,
                "cx": cx,
                "cy": cy,
                "bbox": (card_x1, card_y1, card_x2 - card_x1, card_y2 - card_y1)
            })

    print("\nOCR Method Detections:")
    for card in ocr_detected_cards:
        print(f"  {card['rank']}{card['suit'][0].upper()} at cx={card['cx']:.1f}, cy={card['cy']:.1f}")

    print("\n=========================================")
    print("METHOD 2: Template-Matching Card Detection")
    print("=========================================")

    # Load templates
    templates_dir = "templates"
    ranks = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
    rank_templates = {}
    for r in ranks:
        p = os.path.join(templates_dir, f"rank_{r}.png")
        if os.path.exists(p):
            rank_templates[r] = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        else:
            p2 = os.path.join(templates_dir, f"{r}.png")
            if os.path.exists(p2):
                rank_templates[r] = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)

    suits = ["spade", "heart", "diamond", "club"]
    suit_templates = {}
    for s in suits:
        p = os.path.join(templates_dir, f"{s}.png")
        if os.path.exists(p):
            suit_templates[s] = cv2.imread(p, cv2.IMREAD_GRAYSCALE)

    # Match ranks
    rank_detections = []
    for r, tpl in rank_templates.items():
        res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
        h_tpl, w_tpl = tpl.shape
        threshold = 0.65
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            score = res[pt[1], pt[0]]
            rank_detections.append((pt[0], pt[1], w_tpl, h_tpl, score, r))
            
    filtered_ranks = nms(rank_detections, threshold=0.3)
    filtered_ranks.sort(key=lambda x: (x[1], x[0]))

    # Match suits
    suit_detections = []
    for s, tpl in suit_templates.items():
        res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
        h_tpl, w_tpl = tpl.shape
        threshold = 0.65
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            score = res[pt[1], pt[0]]
            suit_detections.append((pt[0], pt[1], w_tpl, h_tpl, score, s))

    filtered_suits = nms(suit_detections, threshold=0.3)
    filtered_suits.sort(key=lambda x: (x[1], x[0]))

    # Pair ranks and suits using spatial relationships
    # suit is below rank: sx - rx in [3, 12], sy - ry in [22, 38]
    paired_cards = []
    used_suits = set()
    used_ranks = set()

    for r_idx, (rx, ry, rw, rh, r_score, r_label) in enumerate(filtered_ranks):
        best_suit = None
        best_suit_idx = -1
        min_dist = float('inf')
        
        for s_idx, (sx, sy, sw, sh, s_score, s_label) in enumerate(filtered_suits):
            if s_idx in used_suits:
                continue
            
            dx = sx - rx
            dy = sy - ry
            
            # Check geometric constraint
            if 3 <= dx <= 13 and 22 <= dy <= 38:
                dist = np.sqrt(dx**2 + dy**2)
                if dist < min_dist:
                    min_dist = dist
                    best_suit = (sx, sy, sw, sh, s_score, s_label)
                    best_suit_idx = s_idx
                    
        if best_suit:
            paired_cards.append({
                "rank": r_label,
                "suit": best_suit[5],
                "rx": rx, "ry": ry,
                "sx": best_suit[0], "sy": best_suit[1],
                "r_score": r_score, "s_score": best_suit[4]
            })
            used_suits.add(best_suit_idx)
            used_ranks.add(r_idx)

    # Let's print out the paired cards
    print("\nPaired Cards (Template Matching):")
    for pc in paired_cards:
        print(f"  {pc['rank']}{pc['suit'][0].upper()} | Rank score: {pc['r_score']:.3f}, Suit score: {pc['s_score']:.3f} | rx: {pc['rx']}, ry: {pc['ry']} | sx: {pc['sx']}, sy: {pc['sy']}")

    # Let's see if there are unpaired ranks or suits that are highly confident
    print("\nUnpaired Ranks:")
    for r_idx, r in enumerate(filtered_ranks):
        if r_idx not in used_ranks:
            print(f"  Rank: {r[5]:<2} | score: {r[4]:.3f} | x: {r[0]:<3} | y: {r[1]:<3}")

    print("\nUnpaired Suits:")
    for s_idx, s in enumerate(filtered_suits):
        if s_idx not in used_suits:
            print(f"  Suit: {s[5]:<8} | score: {s[4]:.3f} | x: {s[0]:<3} | y: {s[1]:<3}")

    # Create copy of image to draw Template Matching annotations
    img_tmpl = img.copy()
    for pc in paired_cards:
        # Draw Rank bounding box (green)
        cv2.rectangle(img_tmpl, (pc['rx'], pc['ry']), (pc['rx'] + 18, pc['ry'] + 26), (0, 255, 0), 1)
        # Draw Suit bounding box (blue)
        cv2.rectangle(img_tmpl, (pc['sx'], pc['sy']), (pc['sx'] + 13, pc['sy'] + 13), (255, 0, 0), 1)
        # Draw label
        label = f"{pc['rank']}{pc['suit'][0].upper()}"
        cv2.putText(img_tmpl, label, (pc['rx'], pc['ry'] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

    # Save individual annotation images
    cv2.imwrite("debug/annotated_ocr.png", img_ocr)
    cv2.imwrite("debug/annotated_tmpl.png", img_tmpl)

    # Create side-by-side comparison image
    comparison = np.zeros((h_img, w_img * 2 + 10, 3), dtype=np.uint8)
    comparison[:, :w_img] = img_ocr
    comparison[:, w_img:w_img+10] = (50, 50, 50) # separator
    comparison[:, w_img+10:] = img_tmpl

    # Draw titles
    cv2.putText(comparison, "OCR-based (main.py)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(comparison, "Template Matching", (w_img + 20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    comp_path = os.path.join(artifact_dir, "comparison.png")
    cv2.imwrite(comp_path, comparison)
    print(f"\nSaved comparison image to {comp_path}")

if __name__ == "__main__":
    main()
