import cv2
import pytesseract
import csv
from io import StringIO
import os

def test_ocr_thresholds():
    img = cv2.imread("debug_captures/1_ui_full.png")
    if img is None:
        print("Error: debug_captures/1_ui_full.png not found")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Try different thresholds to segment cards (which are white)
    # Since cards are white (intensity > 200), we threshold at different values.
    # We want white card background (255) and black text (0) or vice versa.
    for thresh_val in [150, 180, 200, 220]:
        _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        # Now thresh has 255 (white card body) and 0 (background felt and black/red text).
        # To make it black text on white background globally:
        # The background felt is black (0), which Tesseract might treat as a border or text.
        # What if we invert it, or mask it?
        # Actually, let's try OCR directly on thresh, and also on cv2.bitwise_not(thresh)
        for invert in [False, True]:
            proc = cv2.bitwise_not(thresh) if invert else thresh
            
            # Scale up
            fx = 3.0
            scaled = cv2.resize(proc, (0, 0), fx=fx, fy=fx, interpolation=cv2.INTER_NEAREST)
            
            for psm in [11, 12, 6]:
                config = f"--psm {psm}"
                data_str = pytesseract.image_to_data(scaled, config=config, output_type=pytesseract.Output.STRING)
                
                f = StringIO(data_str)
                reader = csv.reader(f, delimiter='\t')
                header = next(reader)
                
                left_idx = header.index('left')
                top_idx = header.index('top')
                width_idx = header.index('width')
                height_idx = header.index('height')
                text_idx = header.index('text')
                conf_idx = header.index('conf')
                
                found = []
                for row in reader:
                    if len(row) <= text_idx:
                        continue
                    text = row[text_idx].strip()
                    if not text:
                        continue
                    conf = float(row[conf_idx])
                    if conf < 10:
                        continue
                    cx = (int(row[left_idx]) + int(row[width_idx]) // 2) / fx
                    cy = (int(row[top_idx]) + int(row[height_idx]) // 2) / fx
                    found.append((text, conf, cx, cy))
                    
                card_ranks = ["A", "K", "Q", "J", "T", "10", "9", "8", "7", "6", "5", "4", "3", "2"]
                detected = [item for item in found if item[0].upper() in card_ranks]
                
                if len(detected) > 0:
                    print(f"\nThresh={thresh_val}, Invert={invert}, PSM={psm} -> Found {len(detected)} rank characters:")
                    for text, conf, cx, cy in detected[:20]:
                        print(f"  {text!r:5} | Conf: {conf:5.1f}% | cx={cx:.1f}, cy={cy:.1f}")

if __name__ == "__main__":
    test_ocr_thresholds()
