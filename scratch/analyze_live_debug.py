import cv2
import numpy as np
import os
import pytesseract
from analyzer import BridgeAnalyzer

def analyze_image(filepath):
    print(f"\n=========================================")
    print(f"ANALYZING: {filepath}")
    print(f"=========================================")
    img = cv2.imread(filepath)
    if img is None:
        print("❌ Failed to load image")
        return
        
    a = BridgeAnalyzer(verbose=True)
    
    h_strip = img.shape[0]
    w_strip = img.shape[1]
    
    hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    card_mask = (hsv_full[:, :, 1] < 50) & (hsv_full[:, :, 2] > 170)
    row_card_counts = np.sum(card_mask, axis=1)
    card_rows = np.where(row_card_counts > 0.05 * w_strip)[0]
    
    if len(card_rows) >= 10:
        y_start = card_rows[0]
        y_end = card_rows[-1]
        y_start = max(0, y_start - 2)
        y_end = min(h_strip - 1, y_end + 2)
        img_cropped = img[y_start:y_end+1, :]
        h_strip = img_cropped.shape[0]
    else:
        img_cropped = img.copy()

    scale = 60.0 / h_strip
    hand_img = cv2.resize(img_cropped, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    w_strip = hand_img.shape[1]
    
    hsv = cv2.cvtColor(hand_img, cv2.COLOR_BGR2HSV)
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
    min_dist = 15
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
                    peaks.append({"x_suit": x, "color": color})

    # Group consecutive peaks of the same color
    groups = []
    for p in peaks:
        if not groups or p["color"] != groups[-1]["color"]:
            groups.append({"color": p["color"], "peaks": [p]})
        else:
            groups[-1]["peaks"].append(p)
            
    black_count = sum(1 for g in groups if g["color"] == "BLACK")
    red_count = sum(1 for g in groups if g["color"] == "RED")
    black_idx = 0
    red_idx = 0
    
    for g in groups:
        if g["color"] == "BLACK":
            if black_count >= 2:
                suit_name = "spade" if black_idx == 0 else "club"
            else:
                avg_x = sum(p["x_suit"] for p in g["peaks"]) / len(g["peaks"])
                suit_name = "spade" if avg_x < (w_strip / 2) else "club"
            black_idx += 1
        else:
            if red_count >= 2:
                suit_name = "heart" if red_idx == 0 else "diamond"
            else:
                avg_x = sum(p["x_suit"] for p in g["peaks"]) / len(g["peaks"])
                suit_name = "heart" if avg_x < (w_strip / 2) else "diamond"
            red_idx += 1
        for p in g["peaks"]:
            p["assigned_suit"] = suit_name

    print(f"Found {len(peaks)} peaks: {[p['x_suit'] for p in peaks]}")
    
    for idx, p in enumerate(peaks):
        x_card = max(0, p["x_suit"] - 15)
        card_crop = hand_img[0:60, x_card:min(x_card + 40, w_strip)]
        suit = p.get("assigned_suit")
        
        # Local OCR
        local_r, _ = a.extract_card(card_crop)
        candidates = a.extract_card_candidates(card_crop)
        
        # Template Matching Scores on all 13 ranks
        scores = {}
        rank_crop = card_crop[2:30, 2:36]
        gray_crop = cv2.cvtColor(rank_crop, cv2.COLOR_BGR2GRAY)
        for rank in ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]:
            scores[rank] = a.score_rank_candidate(rank_crop, rank)
            
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_scores_str = ", ".join([f"{r}:{s:.3f}" for r, s in sorted_scores[:4]])
        
        print(f"Card {idx:2d} (x={p['x_suit']}, suit={suit}):")
        print(f"  Local OCR: {local_r} | Candidates: {candidates}")
        print(f"  Top templates: {top_scores_str}")

def main():
    analyze_image("debug/player_hand_area.png")
    analyze_image("debug/dummy_strip_north.png")

if __name__ == "__main__":
    main()
