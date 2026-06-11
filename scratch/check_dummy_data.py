import cv2
import pytesseract
import csv
from io import StringIO

def inspect():
    img = cv2.imread("debug_captures/1_ui_full.png")
    if img is None:
        print("Error loading image")
        return
        
    crop = img[275:310, 0:510]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    fx = 4.0
    scaled = cv2.resize(gray, (0, 0), fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
    thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    proc = cv2.bitwise_not(thresh)
    
    data_str = pytesseract.image_to_data(proc, config="--psm 6", output_type=pytesseract.Output.STRING)
    
    f = StringIO(data_str)
    reader = csv.reader(f, delimiter='\t')
    header = next(reader)
    
    text_idx = header.index('text')
    left_idx = header.index('left')
    top_idx = header.index('top')
    width_idx = header.index('width')
    height_idx = header.index('height')
    
    print("--- Detected Text in Compact Dummy Crop ---")
    for row in reader:
        if len(row) <= text_idx:
            continue
        text = row[text_idx].strip()
        if not text:
            continue
        
        # Scale coordinates back to original size (1x)
        left = float(row[left_idx]) / fx
        top = float(row[top_idx]) / fx
        w = float(row[width_idx]) / fx
        h = float(row[height_idx]) / fx
        
        print(f"Text: {text!r:15} | Left: {left:6.1f} | Width: {w:5.1f} | Top: {top:5.1f} | Height: {h:5.1f}")

if __name__ == "__main__":
    inspect()
