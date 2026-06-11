import cv2
import numpy as np
from collections import Counter
import pytesseract

def clean_rank_candidate(text):
    t = text.strip().upper().replace(" ", "")
    if not t:
        return None
    if t == "10":
        return "T"
    if t == "1":
        return "T"
    if t in ["0", "O", "D"]:
        return "Q"
    valid = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
    return t if t in valid else None

def inspect():
    img = cv2.imread("debug_captures/1_ui_full.png")
    if img is None:
        print("Could not load image")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if y < 250 and cv2.contourArea(c) > 100 and w < 300 and h < 110:
            candidates.append((x, y, w, h))
            
    candidates.sort()
    print("Candidates (contours under y=250):")
    for idx, (x, y, w, h) in enumerate(candidates):
        print(f"  Contour {idx}: x={x}, y={y}, w={w}, h={h}")
        
if __name__ == "__main__":
    inspect()
