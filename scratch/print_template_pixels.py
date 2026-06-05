import cv2
import os
import numpy as np

def main():
    templates_dir = "templates"
    for suit in ["spade", "heart", "diamond", "club"]:
        path = os.path.join(templates_dir, f"{suit}.png")
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            print(f"\n--- Template '{suit}' (shape: {img.shape}) ---")
            for r in range(img.shape[0]):
                row_str = "".join(["#" if val > 127 else " " for val in img[r]])
                print(row_str)
        else:
            print(f"Template '{suit}': NOT FOUND")

if __name__ == "__main__":
    main()
