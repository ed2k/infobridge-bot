import cv2
import numpy as np

def main():
    img = cv2.imread("debug_captures/live_ui_all_sides.png")
    if img is None:
        print("Error")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("Contours with area > 100 in y < 250:")
    for idx, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if y < 250 and area > 100:
            print(f"Index {idx}: x={x}..{x+w} (w={w}), y={y}..{y+h} (h={h}), area={area:.1f}")

if __name__ == "__main__":
    main()
