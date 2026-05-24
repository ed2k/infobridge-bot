import cv2
import os
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
    # Load the full UI image (relative y = 135 to 255)
    ui_img_path = "/Users/admin/Documents/GitHub/infobridge-bot/debug_captures/1_ui_full.png"
    if not os.path.exists(ui_img_path):
        print("Error: 1_ui_full.png not found")
        return
        
    ui_img = cv2.imread(ui_img_path)
    
    # Crop using the original config boundaries (y_offset=135, height=120)
    rel_x = 106
    width = 210
    rel_y = 135
    height = 120
    
    crop = ui_img[rel_y:rel_y+height, rel_x:rel_x+width].copy()
    
    # Mask out the player name at the top (top 20 pixels of the crop)
    # We fill it with the table green color (from the top-right corner of the crop)
    bg_color = crop[5, width - 5].tolist() # Get the background green color dynamically
    print(f"Detected background table green color: {bg_color}")
    
    # Fill top 20 pixels with background color
    crop[0:22, :] = bg_color
    
    # Save the masked crop for visual inspection
    out_path = "/Users/admin/Documents/GitHub/infobridge-bot/debug_test_crops/bidding_masked.png"
    cv2.imwrite(out_path, crop)
    print(f"Saved masked crop to {out_path}")
    
    # Run OCR processing
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    processed = cv2.resize(gray, (0, 0), fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
    
    # Run OCR with image_to_data and --psm 6
    data_str = pytesseract.image_to_data(processed, config="--psm 6", output_type=pytesseract.Output.STRING)
    
    print("\n--- OCR words in processed masked image ---")
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
