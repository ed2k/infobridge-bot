import cv2
import numpy as np

def ascii_art(img):
    h, w = img.shape[:2]
    chars = " .:-=+*#%@"
    art = ""
    for r in range(h):
        for c in range(w):
            val = img[r, c]
            idx = int(val / 256.0 * len(chars))
            art += chars[idx]
        art += "\n"
    return art

def main():
    img = cv2.imread("debug_captures/live_ui_all_sides.png")
    if img is None:
        print("Error")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blocks = [
        (19, 75, "Block 1"),
        (142, 75, "Block 2"),
        (279, 75, "Block 3")
    ]
    
    for x, y, name in blocks:
        print(f"\n=== {name} Suit Region (x={x}, y={y}) ===")
        crop = gray[y+5 : y+22, x : x+22]
        print(ascii_art(crop))

if __name__ == "__main__":
    main()
