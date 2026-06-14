import cv2
import numpy as np
import os
import pytesseract
from analyzer import BridgeAnalyzer

def test_hand(filepath):
    print(f"\n--- Testing {os.path.basename(filepath)} ---")
    img = cv2.imread(filepath)
    if img is None:
        print("❌ Failed to load image")
        return
        
    a = BridgeAnalyzer(verbose=True)
    
    # 1. Run full extract_hand_cards (which does global OCR overwrite)
    res_full = a.extract_hand_cards(img)
    full_ranks = [c["rank"] for c in res_full]
    
    # 2. Run local OCR only (by temporarily disabling the global OCR block)
    # We subclass or mock the global OCR check
    # Let's temporarily patch len(detected_cards) check to be negative
    import sys
    orig_extract = a.extract_hand_cards
    
    # We'll run a custom version of extract_hand_cards or inspect the local ranks
    # In analyzer.py, we have detected_cards list returned.
    # Let's inspect the crops directly using the exact same logic.
    h_strip = img.shape[0]
    w_strip = img.shape[1]
    
    hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    card_mask = (hsv_full[:, :, 1] < 50) & (hsv_full[:, :, 2] > 170)
    row_card_counts = np.sum(card_mask, axis=1)
    card_rows = np.where(row_card_counts > 0.05 * w_strip)[0]
    
    y_start_orig = 0
    if len(card_rows) >= 10:
        y_start = card_rows[0]
        y_end = card_rows[-1]
        y_start = max(0, y_start - 2)
        y_end = min(h_strip - 1, y_end + 2)
        img_cropped = img[y_start:y_end+1, :]
        h_strip = img_cropped.shape[0]
        y_start_orig = y_start
    else:
        img_cropped = img.copy()

    # Scale to 60
    scale = 60.0 / h_strip
    hand_img = cv2.resize(img_cropped, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h_strip = 60
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
                    peaks.append({
                        "x_suit": x,
                        "color": color
                    })

    local_ranks = []
    for idx, p in enumerate(peaks):
        x_card = max(0, p["x_suit"] - 15)
        card_crop = hand_img[0:60, x_card:min(x_card + 40, w_strip)]
        suit_left = max(0, p["x_suit"] - 15)
        suit_crop_wide = hand_img[30:55, suit_left:min(suit_left + 40, w_strip)]
        suit_top = hand_img[3:22, suit_left:min(suit_left + 40, w_strip)]
        rank, suit = a.extract_card(card_crop, suit_img=suit_crop_wide, suit_img_top=suit_top, expected_suit_is_red=p["color"] == "RED")
        local_ranks.append(rank)
        
    print(f"Local OCR Ranks Only: {local_ranks}")
    print(f"Global OCR Ranks:     {full_ranks}")
    
    mismatches = sum(1 for l, f in zip(local_ranks, full_ranks) if l and l != f)
    print(f"Mismatches (excluding None): {mismatches}")

def main():
    test_hand("debug/player_hand_area.png")
    test_hand("debug/dummy_strip_north.png")

if __name__ == "__main__":
    main()
