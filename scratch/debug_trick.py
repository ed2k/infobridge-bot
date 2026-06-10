import cv2
import numpy as np
from analyzer import BridgeAnalyzer

def main():
    img = cv2.imread("debug_captures/3_trick.png")
    analyzer = BridgeAnalyzer(verbose=True)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    cv2.imwrite("/Users/admin/.gemini/antigravity-ide/brain/99eec12b-56cc-4eb2-b9c6-920fc66c5416/trick_thresh.png", thresh)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("Found contours:", len(contours))
    
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w)/h
        print(f"Contour {i}: area={area}, bbox=({x},{y},{w},{h}), aspect_ratio={aspect_ratio:.2f}")
        
        if area > 1000 and 0.3 < aspect_ratio < 1.8:
            card_crop = img[y:y+h, x:x+w]
            cv2.imwrite(f"/Users/admin/.gemini/antigravity-ide/brain/99eec12b-56cc-4eb2-b9c6-920fc66c5416/card_crop_{i}.png", card_crop)
            rank, suit = analyzer.extract_card(card_crop)
            print(f"  -> Rank={rank}, Suit={suit}")

if __name__ == "__main__":
    main()
