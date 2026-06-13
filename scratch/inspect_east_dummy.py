import cv2
import numpy as np
import os
import pytesseract
from io import StringIO
import csv

def main():
    east_path = "debug/dummy_strip_east.png"
    if not os.path.exists(east_path):
        print(f"❌ {east_path} not found.")
        return

    img = cv2.imread(east_path)
    print(f"dummy_strip_east.png shape: {img.shape}")

    # Print shapes of cropped cards
    print("\nCropped card files in debug/:")
    for f in sorted(os.listdir("debug")):
        if f.startswith("dummy_card_East_"):
            p = os.path.join("debug", f)
            c_img = cv2.imread(p)
            print(f"  {f}: shape={c_img.shape}")

    # Let's run OCR on the strip
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for thresh_val, invert, psm in [(200, True, 11)]:
        _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        proc = cv2.bitwise_not(thresh) if invert else thresh
        scaled = cv2.resize(proc, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_NEAREST)
        
        try:
            data_str = pytesseract.image_to_data(scaled, config=f"--psm {psm}", output_type=pytesseract.Output.STRING)
            print(f"\nOCR Results (psm={psm}, thresh={thresh_val}):")
            f_csv = StringIO(data_str)
            reader = csv.reader(f_csv, delimiter='\t')
            header = next(reader)
            print(f"Header: {header}")
            for row in reader:
                if len(row) > 11 and row[11].strip():
                    print(f"  Text: {row[11]:<10} | conf: {row[10]:<5} | left: {row[6]:<5} | top: {row[7]:<5} | w: {row[8]:<5} | h: {row[9]:<5}")
        except Exception as e:
            print(f"OCR Error: {e}")

if __name__ == "__main__":
    main()
