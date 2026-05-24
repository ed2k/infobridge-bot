import cv2
import os
import json
import pytesseract
from PIL import Image

def main():
    ui_img_path = "/Users/admin/Documents/GitHub/infobridge-bot/debug_captures/1_ui_full.png"
    if not os.path.exists(ui_img_path):
        print(f"Error: {ui_img_path} not found")
        return
        
    ui_img = cv2.imread(ui_img_path)
    h_ui, w_ui = ui_img.shape[:2]
    print(f"UI Image size: {w_ui}x{h_ui}")
    
    # Current config.json values
    # ui_roi: x=1245, y=220, w=435, h=830
    # bidding_roi: x=1351, y=355, w=210, h=120
    # Rel x = 1351 - 1245 = 106
    # Rel y = 355 - 220 = 135
    
    # We will test different rel_y values (e.g., from 110 to 140) and heights
    test_params = [
        {"y_offset": 150, "height": 105, "label": "offset_y150_h105"},
        {"y_offset": 152, "height": 103, "label": "offset_y152_h103"},
        {"y_offset": 154, "height": 101, "label": "offset_y154_h101"},
        {"y_offset": 156, "height": 99, "label": "offset_y156_h99"},
        {"y_offset": 158, "height": 97, "label": "offset_y158_h97"},
    ]
    
    os.makedirs("/Users/admin/Documents/GitHub/infobridge-bot/debug_test_crops", exist_ok=True)
    
    for param in test_params:
        rel_x = 106
        width = 210
        rel_y = param["y_offset"]
        height = param["height"]
        
        # Crop from UI image
        crop = ui_img[rel_y:rel_y+height, rel_x:rel_x+width]
        out_path = f"/Users/admin/Documents/GitHub/infobridge-bot/debug_test_crops/bidding_{param['label']}.png"
        cv2.imwrite(out_path, crop)
        print(f"Saved crop to {out_path}")
        
        # Run OCR test
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        processed = cv2.resize(gray, (0, 0), fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
        
        # Let's run simple OCR and print the output
        text = pytesseract.image_to_string(processed, config="--psm 6")
        print(f"--- OCR Output for {param['label']} ---")
        print(text)
        print("------------------------------------------\n")

if __name__ == "__main__":
    main()
