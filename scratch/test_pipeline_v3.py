import cv2
import numpy as np
import pytesseract
from analyzer import BridgeAnalyzer

def test_pipeline(img_path, offset):
    print(f"\n=========================================")
    print(f"TESTING PIPELINE: {img_path} | Offset={offset}")
    print(f"=========================================")
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Failed to load image")
        return
        
    a = BridgeAnalyzer(verbose=False)
    
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
    
    # Peak detection with boundary fix
    peaks = []
    min_dist = 15
    for x in range(0, len(smoothed)):
        val = smoothed[x]
        if val >= 2.0:
            is_max = True
            for dx in range(-min_dist, min_dist + 1):
                nx = x + dx
                if 0 <= nx < len(smoothed):
                    if smoothed[nx] > val:
                        is_max = False
                        break
            if is_max:
                if not peaks or (x - peaks[-1]) >= min_dist:
                    peaks.append(x)

    print(f"Peaks found ({len(peaks)}): {peaks}")
    
    if "player_hand_area.png" in img_path:
        true_ranks = ["K", "Q", "7", "5", "2", "6", "9", "5", "K", "J", "T", "8", "2"] # 13 cards!
    else:
        # North Dummy true ranks
        true_ranks = ["A", "K", "J", "5", "7", "5", "2", "Q", "8", "7", "6", "5", "4", "T"] # 14 cards?
        # Wait, let's verify if North dummy has 13 or 14 cards.
        
    for idx, peak in enumerate(peaks):
        x_card = max(0, peak - offset)
        card_crop = hand_img[0:60, x_card:min(x_card + 40, w_strip)]
        
        # Local OCR
        local_r, _ = a.extract_card(card_crop)
        true_r = true_ranks[idx] if idx < len(true_ranks) else "?"
        
        # Score
        rank_crop = card_crop[2:30, 2:36]
        score = a.score_rank_candidate(rank_crop, true_r) if true_r != "?" else -1.0
        
        print(f"  Card {idx:2d} (x={peak:3d}): True={true_r} | OCR={local_r} | Score={score:.3f}")

def main():
    test_pipeline("debug/player_hand_area.png", 15)
    test_pipeline("debug/player_hand_area.png", 18)
    test_pipeline("debug/player_hand_area.png", 22)
    test_pipeline("debug/dummy_strip_north.png", 15)
    test_pipeline("debug/dummy_strip_north.png", 18)

if __name__ == "__main__":
    main()
