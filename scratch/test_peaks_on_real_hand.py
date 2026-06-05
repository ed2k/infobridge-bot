import cv2
import numpy as np
import os
from analyzer import BridgeAnalyzer

def run_peak_detection(hand_img, min_dist, score_thresh):
    analyzer = BridgeAnalyzer(verbose=False)
    
    h_strip = hand_img.shape[0]
    w_strip = hand_img.shape[1]
    
    # Scale to height 60
    scale = 60.0 / h_strip
    hand_img = cv2.resize(hand_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h_strip = 60
    w_strip = hand_img.shape[1]
    
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([25, 255, 255])
    lower_red2 = np.array([165, 40, 40])
    upper_red2 = np.array([180, 255, 255])
    
    scores = []
    for x in range(0, w_strip - 15):
        y_start = 37
        y_end = 50
        suit_crop = hand_img[y_start:y_end, x:x+13]
        
        hsv = cv2.cvtColor(suit_crop, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        red_ratio = np.sum(red_mask > 0) / (suit_crop.shape[0] * suit_crop.shape[1])
        is_red = red_ratio > 0.015
        
        allowed_suits = ["heart", "diamond"] if is_red else ["spade", "club"]
        gray_crop = cv2.cvtColor(suit_crop, cv2.COLOR_BGR2GRAY)
        
        best_score = -1.0
        best_suit = None
        for suit in allowed_suits:
            template = analyzer.suit_templates.get(suit)
            if template is not None:
                res = cv2.matchTemplate(gray_crop, template, cv2.TM_CCOEFF_NORMED)
                val = res[0][0]
                if val > best_score:
                    best_score = val
                    best_suit = suit
                    
        scores.append((x, best_score, best_suit))
        
    peaks = []
    for idx in range(1, len(scores) - 1):
        x, score, suit = scores[idx]
        if score > score_thresh:
            is_peak = True
            start_check = max(0, idx - min_dist)
            end_check = min(len(scores), idx + min_dist)
            for j in range(start_check, end_check):
                if scores[j][1] > score:
                    is_peak = False
                    break
            if is_peak:
                if not peaks or (x - peaks[-1]["x_suit"]) >= min_dist:
                    peaks.append({"x_suit": x, "score": score, "suit": suit})
                    
    print(f"min_dist={min_dist}, score_thresh={score_thresh}: Found {len(peaks)} cards:")
    cards_list = []
    for p in peaks:
        # OCR rank
        x_card = max(0, p["x_suit"] - 5)
        card_crop = hand_img[0:60, x_card:min(x_card + 40, w_strip)]
        rank, _ = analyzer.extract_card(card_crop)
        cards_list.append(f"{rank or '?'}{p['suit']}")
    print("  " + ", ".join(cards_list))

def main():
    img_path = "debug_captures/1_ui_full.png"
    if not os.path.exists(img_path):
        print("❌ UI image not found.")
        return
        
    img = cv2.imread(img_path)
    hand_crop = img[651:716, 15:433]
    
    run_peak_detection(hand_crop, min_dist=10, score_thresh=0.38)
    run_peak_detection(hand_crop, min_dist=12, score_thresh=0.38)
    run_peak_detection(hand_crop, min_dist=13, score_thresh=0.35)

if __name__ == "__main__":
    main()
