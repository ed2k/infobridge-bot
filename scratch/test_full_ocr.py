import cv2
import pytesseract
import csv
from io import StringIO
from analyzer import BridgeAnalyzer

def test_ocr(img, psm, use_whitelist):
    fx = 2.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, (0, 0), fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
    
    config = f"--psm {psm}"
    if use_whitelist:
        config += " -c tessedit_char_whitelist=AKQJT1098765432"
        
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
    
    results = []
    for row in reader:
        if len(row) <= text_idx:
            continue
        text = row[text_idx].strip()
        conf = float(row[conf_idx])
        if not text or conf < 10:
            continue
            
        x = int(int(row[left_idx]) / fx)
        y = int(int(row[top_idx]) / fx)
        w = int(int(row[width_idx]) / fx)
        h = int(int(row[height_idx]) / fx)
        results.append((text, conf, x, y, w, h))
    return results

def main():
    img = cv2.imread("debug_captures/1_ui_full.png")
    
    for psm in [11, 6, 12]:
        for wl in [True, False]:
            res = test_ocr(img, psm, wl)
            print(f"\n--- PSM={psm}, Whitelist={wl} (Found {len(res)} items) ---")
            # Print the first 20 items
            for item in res[:20]:
                print(f"  {item[0]!r:10} | Conf: {item[1]:5.1f}% | Box: x={item[2]}, y={item[3]}, w={item[4]}, h={item[5]}")

if __name__ == "__main__":
    main()
