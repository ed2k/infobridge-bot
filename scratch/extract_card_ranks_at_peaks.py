import cv2
import numpy as np
import os
import pytesseract
from analyzer import BridgeAnalyzer

def main():
    img_path = "debug_captures/1_ui_full.png"
    if not os.path.exists(img_path):
        print("❌ UI image not found.")
        return
        
    img = cv2.imread(img_path)
    hand_crop = img[651:716, 15:433]
    h_strip = hand_crop.shape[0]
    
    scale = 60.0 / h_strip
    hand_crop = cv2.resize(hand_crop, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    analyzer = BridgeAnalyzer(verbose=False)
    
    peaks = [11, 16, 46, 51, 82, 87, 121, 156, 192, 227, 263, 298, 334, 370]
    
    print("\n--- Card crops at all 14 peaks ---")
    for idx, x_val in enumerate(peaks):
        x_card = max(0, x_val - 5)
        card_crop = hand_crop[0:60, x_card:min(x_card + 40, hand_crop.shape[1])]
        
        # Crop Rank: top-left corner
        rank_crop = card_crop[9:35, 4:18]
        
        # Check if rank crop is mostly white
        gray_rank = cv2.cvtColor(rank_crop, cv2.COLOR_BGR2GRAY)
        dark_pixels = np.sum(gray_rank < 200)
        
        # OCR
        proc_rank = analyzer.preprocess_for_ocr(rank_crop)
        ocr_results = []
        for psm in [8, 10]:
            config = f"--psm {psm} -c tessedit_char_whitelist=AKQJT1098765432"
            raw = pytesseract.image_to_string(proc_rank, config=config).strip().upper()
            if raw:
                ocr_results.append(f"psm{psm}:{raw}")
                
        print(f"Peak {idx+1:2d} at x={x_val:3d} | Dark pixels={dark_pixels} | OCR: {ocr_results}")

if __name__ == "__main__":
    main()
