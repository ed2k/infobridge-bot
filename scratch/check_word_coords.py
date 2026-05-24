import cv2
import pytesseract
import csv
from io import StringIO

def clean_header_text(text):
    import re
    cleaned = re.sub(r'[^a-zA-Z]', '', text).upper()
    if not cleaned:
        return None
    if "SOUTH" in cleaned or cleaned.startswith("S"):
        return "S"
    if "WEST" in cleaned or cleaned.startswith("W"):
        return "W"
    if "NORTH" in cleaned or cleaned.startswith("N"):
        return "N"
    if "EAST" in cleaned or cleaned.startswith("E"):
        return "E"
    return None

def main():
    img_path = "/Users/admin/Documents/GitHub/infobridge-bot/debug_captures/2_bidding.png"
    img = cv2.imread(img_path)
    
    # Scale up
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed = cv2.resize(gray, (0, 0), fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
    
    # Run OCR with image_to_data and --psm 6
    data_str = pytesseract.image_to_data(processed, config="--psm 6", output_type=pytesseract.Output.STRING)
    
    print("--- OCR words in processed image ---")
    f = StringIO(data_str)
    reader = csv.reader(f, delimiter='\t')
    header = next(reader)
    text_idx = header.index('text')
    left_idx = header.index('left')
    top_idx = header.index('top')
    width_idx = header.index('width')
    height_idx = header.index('height')
    
    for row in reader:
        if len(row) <= text_idx:
            continue
        text = row[text_idx].strip()
        if not text:
            continue
        
        left = int(row[left_idx])
        top = int(row[top_idx])
        w = int(row[width_idx])
        h = int(row[height_idx])
        
        dir_key = clean_header_text(text)
        print(f"Text: {text!r:15} | Cleaned: {dir_key!r:5} | Left: {left:4} | Top: {top:4} | Width: {w:4} | Height: {h:4}")

if __name__ == "__main__":
    main()
