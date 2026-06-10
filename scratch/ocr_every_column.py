import cv2
import pytesseract
import numpy as np

def clean_rank(text):
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

def main():
    img = cv2.imread("debug_captures/live_ui_all_sides.png")
    if img is None:
        print("Error")
        return
        
    # We want to scan the three blocks:
    # Block 1: x=19..133, y=75..160
    # Block 2: x=142..256, y=75..160
    # Block 3: x=279..379, y=75..160
    
    # Let's scan x from 19 to 380, at y=75..105 (rank region is top of the card)
    # Card rank is usually in the top-left of each card.
    # Let's crop a window of width 22, height 26, and slide it by 1 pixel.
    h_win, w_win = 26, 22
    y_top = 75 + 2 # rank top offset
    
    print("Sliding window OCR scan:")
    detected = []
    
    for x in range(19, 380 - w_win):
        crop = img[y_top : y_top+h_win, x : x+w_win]
        
        # Preprocess
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        scaled = cv2.resize(gray, (0, 0), fx=5.0, fy=5.0, interpolation=cv2.INTER_CUBIC)
        thresh = cv2.threshold(scaled, 127, 255, cv2.THRESH_BINARY)[1]
        
        # Run OCR
        config = "--psm 10 -c tessedit_char_whitelist=AKQJT1098765432"
        try:
            txt = pytesseract.image_to_string(thresh, config=config)
            r = clean_rank(txt)
            if r:
                detected.append((x, r))
        except Exception:
            pass

    # Filter consecutive detections of the same rank to find peaks
    print("\nRaw detections:")
    for x, r in detected:
        print(f"  x={x}: {r}")

if __name__ == "__main__":
    main()
