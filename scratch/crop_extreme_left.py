import cv2
import numpy as np
import os
import mss
from analyzer import BridgeAnalyzer

def detect_peaks(hand_crop, min_dist=15):
    analyzer = BridgeAnalyzer(verbose=False)
    
    hsv = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2HSV)
    
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([25, 255, 255])
    lower_red2 = np.array([165, 40, 40])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    mask_black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 60, 100]))
    mask_suit = mask_red + mask_black
    
    profile = np.sum(mask_suit[41:54, :] > 0, axis=0).astype(np.float32)
    
    kernel = np.ones(13) / 13.0
    smoothed = np.convolve(profile, kernel, mode='same')
    
    peaks = []
    for x in range(min_dist, len(smoothed) - min_dist):
        val = smoothed[x]
        if val >= 2.0:
            is_max = True
            for dx in range(-min_dist, min_dist + 1):
                if smoothed[x + dx] > val:
                    is_max = False
                    break
            if is_max:
                if not peaks or (x - peaks[-1]["x_suit"]) >= min_dist:
                    col_red = np.sum(mask_red[41:54, x] > 0)
                    col_black = np.sum(mask_black[41:54, x] > 0)
                    color = "RED" if col_red >= col_black else "BLACK"
                    peaks.append({
                        "x_suit": x,
                        "color": color
                    })
    return peaks, hand_crop

def test_extract_card_centered(analyzer, card_img, card_idx):
    h, w = card_img.shape[:2]
    if w < 10 or h < 10:
        return None, None
        
    # Since we cropped x_card = x_suit - 22, the center of suit is at x = 22, and rank is centered at 4.5.
    # We crop the rank area: from 1 to 16 (width 15) to exclude the left border line
    rank_crop = card_img[9:35, 1:16]
    
    # We crop the suit area: from 13 to 31 (width 18)
    suit_crop = card_img[24:55, 13:31]
    
    if rank_crop.size == 0 or suit_crop.size == 0:
        return None, None
        
    # Extract Rank
    processed_rank = analyzer.preprocess_for_ocr(rank_crop)

    def normalize_rank_text(raw_text):
        rank_text = raw_text.strip().upper().replace(" ", "")
        if not rank_text:
            return None

        # Tesseract often reads 10 as 1.
        if "10" in rank_text:
            return "10"
        if rank_text == "1":
            return "10"

        # Common Queen misreads from curved glyphs.
        if rank_text in ["0", "O", "D"]:
            return "Q"

        valid_ranks = ["A", "K", "Q", "J", "T", "10", "9", "8", "7", "6", "5", "4", "3", "2"]
        return rank_text if rank_text in valid_ranks else None

    rank_text = None
    for psm in [8, 10]:
        custom_config = f"--psm {psm} -c tessedit_char_whitelist=AKQJT1098765432"
        raw_rank = pytesseract_to_str(processed_rank, custom_config)
        rank_text = normalize_rank_text(raw_rank)
        if rank_text:
            break
        
    # Extract Suit using template matching or fallback color-shape
    suit = None
    hsv = cv2.cvtColor(suit_crop, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([25, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    red_ratio = np.sum(red_mask > 0) / (suit_crop.shape[0] * suit_crop.shape[1])
    is_red = red_ratio > 0.015
    allowed_suits = ["heart", "diamond"] if is_red else ["spade", "club"]
    
    gray = cv2.cvtColor(suit_crop, cv2.COLOR_BGR2GRAY)
    scores = {}
    for s in allowed_suits:
        template = analyzer.suit_templates.get(s)
        if template is not None:
            t_h, t_w = template.shape[:2]
            g_h, g_w = gray.shape[:2]
            if g_h < t_h or g_w < t_w:
                gray_search = cv2.resize(gray, (max(g_w, t_w), max(g_h, t_h)))
            else:
                gray_search = gray
            res = cv2.matchTemplate(gray_search, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            scores[s] = max_val
            
    best_suit = None
    if scores:
        best_suit = max(scores, key=scores.get)
        if scores[best_suit] > 0.35:
            suit = best_suit
            
    print(f"  [Card {card_idx}] is_red={is_red} scores={scores} chosen_template={suit}")
    
    fallback = analyzer.classify_suit_by_color_shape(suit_crop)
    print(f"  [Card {card_idx}] fallback shape chosen={fallback}")
    
    if not suit:
        suit = fallback
        
    return rank_text, suit

import pytesseract
def pytesseract_to_str(img, config):
    try:
        return pytesseract.image_to_string(img, config=config)
    except Exception:
        return ""

def main():
    monitor = {
        "top": 871,
        "left": 1170,
        "width": 520,
        "height": 65
    }
    
    with mss.mss() as sct:
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        hand_crop = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
    analyzer = BridgeAnalyzer(verbose=False)
    
    peaks, _ = detect_peaks(hand_crop, min_dist=15)
    print(f"\n--- Testing Centered Card Crops ---")
    print(f"Found {len(peaks)} card peaks:")
    cards_list = []
    for idx, p in enumerate(peaks):
        # Center the crop at p["x_suit"]: x_card goes from p["x_suit"] - 22 to p["x_suit"] + 18 (width 40)
        x_card = max(0, p["x_suit"] - 22)
        card_crop = hand_crop[0:60, x_card:min(x_card + 40, hand_crop.shape[1])]
        
        rank, suit = test_extract_card_centered(analyzer, card_crop, idx+1)
        
        if not suit:
            suit = "heart" if p["color"] == "RED" else "spade"
        cards_list.append(f"{rank or '?'}{suit}")
        print(f"  Peak {idx+1:2d} at global x={1170 + p['x_suit']:4d} | Card: {rank or '?'}{suit}")
        
    print("\nFull Cards List:", ", ".join(cards_list))

if __name__ == "__main__":
    main()


