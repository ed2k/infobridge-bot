import cv2
import pytesseract
import csv
from io import StringIO

def main():
    img = cv2.imread("debug_captures/live_ui_all_sides.png")
    if img is None:
        print("Error: debug_captures/live_ui_all_sides.png not found")
        return
        
    top_area = img[0:250, :]
    gray = cv2.cvtColor(top_area, cv2.COLOR_BGR2GRAY)
    
    # Run OCR with PSM 11 and 12
    for psm in [11, 12, 6, 3]:
        print(f"\n--- OCR PSM {psm} ---")
        try:
            config = f"--psm {psm}"
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
                if conf < 0:
                    continue
                x = int(row[left_idx])
                y = int(row[top_idx])
                w = int(row[width_idx])
                h = int(row[height_idx])
                print(f"  [{conf:.1f}] '{text}' at x={x}..{x+w}, y={y}..{y+h}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
