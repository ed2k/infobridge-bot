import cv2
import pytesseract
import csv
from io import StringIO
from analyzer import BridgeAnalyzer

def test_west_ocr():
    img = cv2.imread("debug_captures/1_ui_full.png")
    # West hand relative coordinates:
    # x = 0 to 68, y = 350 to 780
    x, y, w, h = 0, 350, 68, 430
    crop = img[y:y+h, x:x+w]
    
    # Scale up for better OCR
    fx = 4.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, (0, 0), fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
    
    # Run image_to_data
    custom_config = "--psm 11 -c tessedit_char_whitelist=AKQJT1098765432"
    data_str = pytesseract.image_to_data(scaled, config=custom_config, output_type=pytesseract.Output.STRING)
    
    f = StringIO(data_str)
    reader = csv.reader(f, delimiter='\t')
    header = next(reader)
    
    left_idx = header.index('left')
    top_idx = header.index('top')
    width_idx = header.index('width')
    height_idx = header.index('height')
    text_idx = header.index('text')
    conf_idx = header.index('conf')
    
    print("Detected West card rank candidates:")
    for row in reader:
        if len(row) <= text_idx:
            continue
        text = row[text_idx].strip()
        conf = float(row[conf_idx])
        if not text or conf < 30:
            continue
            
        # Coordinates relative to the crop
        cx = int(int(row[left_idx]) / fx)
        cy = int(int(row[top_idx]) / fx)
        cw = int(int(row[width_idx]) / fx)
        ch = int(int(row[height_idx]) / fx)
        
        # Global coordinates in 1_ui_full
        gx = x + cx
        gy = y + cy
        
        print(f"Text: {text!r:5} | Conf: {conf:5.1f}% | Crop Box: x={cx}, y={cy}, w={cw}, h={ch} | Global y={gy}")

if __name__ == "__main__":
    test_west_ocr()
