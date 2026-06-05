import cv2
import os
import pytesseract
import numpy as np
from analyzer import BridgeAnalyzer

def test_on_image(img_path, name):
    if not os.path.exists(img_path):
        print(f"❌ {name} path {img_path} not found.")
        return
    img = cv2.imread(img_path)
    print(f"\n==================================================")
    print(f" TESTING {name}: {img_path} (shape: {img.shape})")
    print(f"==================================================")
    
    analyzer = BridgeAnalyzer(verbose=False)
    
    # Scale if needed to match height 60
    h_strip, w_strip = img.shape[:2]
    scale = 1.0
    if h_strip != 60:
        scale = 60.0 / h_strip
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h_strip = 60
        w_strip = img.shape[1]
        
    # Let's run peak detection or linear slicing to get the card regions
    detected_cards = analyzer.extract_hand_cards(img)
    print(f"Detected {len(detected_cards)} cards using extract_hand_cards.")
    
    for idx, card in enumerate(detected_cards):
        # Let's manually reconstruct the card crop and try different OCR approaches
        # Card bbox is in scaled coordinates
        bbox = card["bbox"] # coordinates at scale 1.0 (which is 60px height)
        x_card = int(bbox["x"] * scale)
        w_card = int(bbox["w"] * scale)
        card_crop = img[0:60, x_card:min(x_card + 40, w_strip)]
        
        # Test original crop: 9:35, 4:18
        rank_crop_orig = card_crop[9:35, 4:18]
        
        # Let's also try a slightly wider/higher crop: e.g. 5:35, 2:22
        rank_crop_wide = card_crop[4:35, 2:22]
        
        # Perform preprocessing
        proc_orig = analyzer.preprocess_for_ocr(rank_crop_orig)
        proc_wide = analyzer.preprocess_for_ocr(rank_crop_wide)
        
        # Save crops for manual inspection
        os.makedirs("debug_ocr_test", exist_ok=True)
        cv2.imwrite(f"debug_ocr_test/{name}_card_{idx}_orig.png", rank_crop_orig)
        cv2.imwrite(f"debug_ocr_test/{name}_card_{idx}_wide.png", rank_crop_wide)
        cv2.imwrite(f"debug_ocr_test/{name}_card_{idx}_proc_orig.png", proc_orig)
        cv2.imwrite(f"debug_ocr_test/{name}_card_{idx}_proc_wide.png", proc_wide)
        
        # Run OCR
        ocr_results = {}
        for label, proc_img in [("orig", proc_orig), ("wide", proc_wide)]:
            for psm in [8, 10, 6, 7]:
                config = f"--psm {psm} -c tessedit_char_whitelist=AKQJT1098765432"
                txt = pytesseract.image_to_string(proc_img, config=config).strip()
                if txt:
                    ocr_results[f"{label}_psm{psm}"] = txt
                    
        print(f"Card {idx+1} (expected suit={card['suit']}):")
        print(f"  OCR results: {ocr_results}")
        print(f"  Final assigned rank: {card['rank']}")

def main():
    test_on_image("sample_board.png", "MOCK")
    test_on_image("debug_captures/4_player_hand.png", "LIVE")

if __name__ == "__main__":
    main()
