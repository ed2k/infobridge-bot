import cv2
import numpy as np
from analyzer import BridgeAnalyzer

def test_extract_card_centered(analyzer, card_img):
    h, w = card_img.shape[:2]
    
    # Let's crop more of the height for the rank (down to 54% of card height)
    rank_crop = card_img[2:int(h*0.54), int(w*0.10):int(w*0.90)]
    suit_crop = card_img[int(h*0.42):int(h*0.95), int(w*0.10):int(w*0.90)]
    
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
        
    return rank_text, suit

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
            rank, suit = test_extract_card_centered(analyzer, card_crop)
            print(f"Contour {i}: bbox=({x},{y},{w},{h}) -> Rank={rank}, Suit={suit}")

if __name__ == "__main__":
    main()
