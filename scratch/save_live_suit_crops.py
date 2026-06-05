import cv2
import numpy as np
import os

def main():
    img_path = "debug_captures/1_ui_full.png"
    if not os.path.exists(img_path):
        print("❌ UI image not found.")
        return
        
    img = cv2.imread(img_path)
    
    # Crop Band 6: relative y 651 to 716, x 15 to 433
    hand_crop = img[651:716, 15:433]
    h_strip = hand_crop.shape[0]
    
    # Scale to height 60
    scale = 60.0 / h_strip
    hand_crop = cv2.resize(hand_crop, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    os.makedirs("debug_ocr_test/live_suits", exist_ok=True)
    cv2.imwrite("debug_ocr_test/live_hand_crop_resized.png", hand_crop)
    
    # Save 13 crops by sliding/splitting or using peak locations
    # Let's save crops of size 13x13 every 32 pixels starting around x=7 to see what they look like
    for i in range(13):
        # Card spacing is roughly 32px or 26px
        # Let's crop a window of 13x13 at y_start=37, y_end=50, width x_start..x_start+13
        x_center = int(7 + i * 32 * scale) # estimation
        x_start = max(0, x_center - 6)
        suit_crop = hand_crop[37:50, x_start:x_start+13]
        cv2.imwrite(f"debug_ocr_test/live_suits/suit_{i}.png", suit_crop)
        
        # Let's also save the entire card crop of size 40x60
        card_crop = hand_crop[0:60, max(0, x_center-10):min(hand_crop.shape[1], x_center+30)]
        cv2.imwrite(f"debug_ocr_test/live_suits/card_{i}.png", card_crop)

    print("Saved live suit crops to debug_ocr_test/live_suits/")

if __name__ == "__main__":
    main()
