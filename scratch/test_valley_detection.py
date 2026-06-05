import cv2
import numpy as np
import os
import pytesseract
from analyzer import BridgeAnalyzer

def detect_hand_cards_valleys(img, name, verbose=True):
    analyzer = BridgeAnalyzer(verbose=False)
    
    h_strip, w_strip = img.shape[:2]
    scale = 60.0 / h_strip
    hand_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h_strip = 60
    w_strip = hand_img.shape[1]
    
    gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
    
    # 1. Compute horizontal brightness profile of the suit row (y = 37..50)
    col_means = np.mean(gray[37:50, :], axis=0)
    
    # 2. Find local minima (valleys)
    valleys = []
    # A column is a valley if it's lower than its neighbors and below a threshold
    # We use a small window size (e.g. radius of 3px) to find valleys that are close
    radius = 3
    threshold = 248.0
    
    for x in range(radius, w_strip - radius):
        val = col_means[x]
        if val < threshold:
            # Check if it's a local minimum in the neighborhood
            is_min = True
            for dx in range(-radius, radius + 1):
                if col_means[x + dx] < val:
                    is_min = False
                    break
            if is_min:
                # Prevent duplicate valleys that are too close (e.g. flat bottom)
                if not valleys or (x - valleys[-1]) > 3:
                    valleys.append(x)
                    
    print(f"\n[{name}] Detected {len(valleys)} valleys at x: {valleys}")
    
    # Let's extract rank and suit for each valley
    detected_cards = []
    for idx, x_val in enumerate(valleys):
        # Card crop width is 40px, starting roughly 5px to the left of the valley center
        x_card = max(0, x_val - 5)
        card_crop = hand_img[0:60, x_card:min(x_card + 40, w_strip)]
        
        # Determine color of suit crop
        suit_crop = hand_img[37:50, x_val-6:x_val+7]
        if suit_crop.shape[1] < 13:
            # Pad if needed
            suit_crop = cv2.copyMakeBorder(suit_crop, 0, 0, 0, 13 - suit_crop.shape[1], cv2.BORDER_CONSTANT, value=[255, 255, 255])
            
        hsv = cv2.cvtColor(suit_crop, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 40, 40]) # slightly lowered saturation/value bounds
        upper_red1 = np.array([25, 255, 255])
        lower_red2 = np.array([165, 40, 40])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_ratio = np.sum(mask1 + mask2 > 0) / (suit_crop.shape[0] * suit_crop.shape[1])
        is_red = red_ratio > 0.015
        
        # Rank extraction
        rank_crop = card_crop[9:35, 4:18]
        if rank_crop.size > 0:
            proc_rank = analyzer.preprocess_for_ocr(rank_crop)
            rank_text = None
            for psm in [8, 10]:
                config = f"--psm {psm} -c tessedit_char_whitelist=AKQJT1098765432"
                raw = pytesseract.image_to_string(proc_rank, config=config).strip().upper()
                if raw in ["A", "K", "Q", "J", "T", "10", "9", "8", "7", "6", "5", "4", "3", "2"]:
                    rank_text = raw
                    break
                # Custom normalization
                if raw == "1" or raw == "10":
                    rank_text = "10"
                    break
                if raw in ["0", "O", "D"]:
                    rank_text = "Q"
                    break
        else:
            rank_text = None
            
        # Suit template matching
        allowed_suits = ["heart", "diamond"] if is_red else ["spade", "club"]
        gray_suit = cv2.cvtColor(suit_crop, cv2.COLOR_BGR2GRAY)
        
        best_suit = None
        best_score = -1.0
        for suit in allowed_suits:
            template = analyzer.suit_templates.get(suit)
            if template is not None:
                res = cv2.matchTemplate(gray_suit, template, cv2.TM_CCOEFF_NORMED)
                score = res[0][0]
                if score > best_score:
                    best_score = score
                    best_suit = suit
                    
        # Fallback to color/shape if match score is extremely low
        if best_score < 0.35:
            best_suit = analyzer.classify_suit_by_color_shape(suit_crop)
            
        detected_cards.append({
            "rank": rank_text,
            "suit": best_suit,
            "x": x_val,
            "is_red": is_red,
            "score": best_score
        })
        
        if verbose:
            print(f"  Card {idx+1:2d} at x={x_val:3d} | is_red={is_red} (ratio={red_ratio:.3f}) | Rank={rank_text or '?'} | Suit={best_suit} (score={best_score:.3f})")

    return detected_cards

def main():
    # Test on Live Band 6
    img_path = "debug_captures/1_ui_full.png"
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        hand_crop = img[651:716, 15:433]
        detect_hand_cards_valleys(hand_crop, "LIVE")
        
    # Test on Mock
    mock_path = "sample_board.png"
    if os.path.exists(mock_path):
        img = cv2.imread(mock_path)
        # Mock hand ROI: y=600, height=60, x=300, width=500. So relative x 15 to 433?
        # In generate_mock.py, player_hand_roi is x=300, y=600, w=500, h=60
        # Wait, the relative coordinates are different. In test_mock.py:
        # hand_crop = img[600:660, 300:800]
        hand_crop = img[600:660, 300:800]
        detect_hand_cards_valleys(hand_crop, "MOCK")

if __name__ == "__main__":
    main()
