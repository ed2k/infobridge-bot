import cv2
import numpy as np
import os

def print_binary(img, thresh=127, name=""):
    print(f"\n--- {name} (shape: {img.shape}) ---")
    for r in range(img.shape[0]):
        row_str = "".join(["#" if val > thresh else "." for val in img[r]])
        print(row_str)

def main():
    template_path = "templates/spade.png"
    live_path = "debug_ocr_test/live_suits/suit_2.png" # card index 2 (a spade)
    
    if os.path.exists(template_path):
        tpl = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        print_binary(tpl, 127, "Template spade")
        # Print actual raw values for a row
        print("Template row 6 raw:", tpl[6])
    else:
        print("❌ Template not found.")
        
    if os.path.exists(live_path):
        live = cv2.imread(live_path, cv2.IMREAD_GRAYSCALE)
        print_binary(live, 200, "Live spade (threshold 200)")
        print_binary(live, 127, "Live spade (threshold 127)")
        # Print actual raw values for a row
        print("Live row 6 raw:", live[6])
    else:
        print("❌ Live crop not found.")

if __name__ == "__main__":
    main()
