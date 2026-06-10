import cv2
import pytesseract
import csv
from io import StringIO

def test_full_ocr():
    img = cv2.imread("debug_captures/1_ui_full.png")
    if img is None:
        print("Error: debug_captures/1_ui_full.png not found")
        return
        
    print(f"Image shape: {img.shape}")
    
    # Let's try OCR on the raw image or resized/preprocessed image
    for fx in [1.0, 2.0, 3.0, 4.0]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if fx != 1.0:
            scaled = cv2.resize(gray, (0, 0), fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
        else:
            scaled = gray
            
        for psm in [11, 12, 6, 3]:
            # No whitelist first, to see raw characters
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
                    
                cx = int(int(row[left_idx]) / fx)
                cy = int(int(row[top_idx]) / fx)
                cw = int(int(row[width_idx]) / fx)
                ch = int(int(row[height_idx]) / fx)
                found.append((text, conf, cx, cy, cw, ch))
                
            print(f"\n=== Scale fx={fx}, PSM={psm} (Found {len(found)} items) ===")
            # Filter and print items that look like card ranks (A, K, Q, J, T, 10, 9-2)
            card_like = []
            for item in found:
                text_clean = item[0].strip().upper()
                if text_clean in ["A", "K", "Q", "J", "T", "10", "9", "8", "7", "6", "5", "4", "3", "2", "0", "1", "O", "D", "S", "H", "C", "X"]:
                    card_like.append(item)
            
            print(f"Card-like items ({len(card_like)}):")
            for item in card_like[:30]:
                print(f"  {item[0]!r:10} | Conf: {item[1]:5.1f}% | Box: x={item[2]}, y={item[3]}, w={item[4]}, h={item[5]}")
            
            print(f"Other items (showing first 15):")
            other = [item for item in found if item not in card_like]
            for item in other[:15]:
                print(f"  {item[0]!r:10} | Conf: {item[1]:5.1f}% | Box: x={item[2]}, y={item[3]}, w={item[4]}, h={item[5]}")

if __name__ == "__main__":
    test_full_ocr()
