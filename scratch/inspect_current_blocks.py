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
    
    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if y < 250 and cv2.contourArea(c) > 100 and w < 300 and h < 110:
            candidates.append((x, y, w, h))
            
    candidates.sort()
    print("Detected North dummy suit blocks in current capture:")
    for idx, (bx, by, bw, bh) in enumerate(candidates):
        card_w = 55
        if bw <= 80:
            num_cards = 1
        else:
            num_cards = int(round((bw - 25 - card_w) / 8.5)) + 1
        print(f"  Block {idx+1}: x={bx}..{bx+bw} (w={bw}), y={by}..{by+bh} (h={bh}) -> Estimated {num_cards} cards")

if __name__ == "__main__":
    main()
