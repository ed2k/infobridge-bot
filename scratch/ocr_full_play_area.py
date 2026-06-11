import cv2
import pytesseract
import csv
from io import StringIO

def run_ocr(gray, psm, title):
    print(f"\n========================================\n{title} (PSM {psm})\n========================================")
    config = f"--psm {psm}"
    try:
        data_str = pytesseract.image_to_data(gray, config=config, output_type=pytesseract.Output.STRING)
        f = StringIO(data_str)
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        
        left_idx = header.index('left')
        top_idx = header.index('top')
        width_idx = header.index('width')
        height_idx = header.index('height')
        text_idx = header.index('text')
        conf_idx = header.index('conf')
        
        for row in reader:
            if len(row) <= text_idx:
                continue
            text = row[text_idx].strip()
            if not text:
                continue
            conf = float(row[conf_idx])
            # Filter out very low confidence
            if conf < 10:
                continue
            x = int(row[left_idx])
            y = int(row[top_idx])
            w = int(row[width_idx])
            h = int(row[height_idx])
            print(f"  [{conf:5.1f}%] '{text}' at x={x}..{x+w}, y={y}..{y+h}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    img = cv2.imread("debug_captures/live_ui_all_sides.png")
    if img is None:
        print("Error: live_ui_all_sides.png not found")
        return
        
    h_img, w_img = img.shape[:2]
    print(f"Image dimensions: {w_img}x{h_img}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Run sparse text OCR (PSM 11)
    run_ocr(gray, 11, "Sparse Text Detection")
    
    # Run structured block OCR (PSM 6)
    run_ocr(gray, 6, "Structured Text Detection")

if __name__ == "__main__":
    main()
