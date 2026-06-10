import cv2
import numpy as np
from analyzer import BridgeAnalyzer

def test_extract_card_centered_dynamic(analyzer, card_img):
    h, w = card_img.shape[:2]
    
    # Crop the top 65% of the card
    top_65 = card_img[0:int(h*0.65), :]
    
    # Convert to grayscale and threshold to binary (inverted: ink is 255)
    gray = cv2.cvtColor(top_65, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Compute horizontal projection
    row_sums = np.sum(binary > 0, axis=1)
    
    # Search for the gap between int(h * 0.40) and int(h * 0.62)
    search_start = int(h * 0.40)
    search_end = int(h * 0.62)
    
    best_y = int(h * 0.50)
    min_sum = float('inf')
    for y in range(search_start, search_end):
        window_sum = np.sum(row_sums[y-1:y+2])
        if window_sum < min_sum:
            min_sum = window_sum
            best_y = y
            
    # Crop rank from 2 to best_y
    rank_crop = card_img[2:best_y, int(w*0.10):int(w*0.90)]
    suit_crop = card_img[best_y:int(h*0.95), int(w*0.10):int(w*0.90)]
    
    # Run OCR on rank
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
                    break
            except Exception:
                pass
        if rank_text:
            break
            
    # Run suit classification
    suit = analyzer.classify_suit_template_matching(suit_crop)
    if not suit:
        suit = analyzer.classify_suit_by_color_shape(suit_crop)
        
    return rank_text, suit, best_y

def main():
    img = cv2.imread("debug_captures/3_trick.png")
    analyzer = BridgeAnalyzer(verbose=True)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w)/h
        
        if 1500 < area < 4000:
            card_crop = img[y:y+h, x:x+w]
            rank, suit, split_y = test_extract_card_centered_dynamic(analyzer, card_crop)
            print(f"Contour {i}: bbox=({x},{y},{w},{h}), split_y={split_y} -> Rank={rank}, Suit={suit}")

if __name__ == "__main__":
    main()
