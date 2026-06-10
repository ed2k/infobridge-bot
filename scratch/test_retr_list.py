import cv2
import numpy as np

def main():
    img = cv2.imread("debug_captures/3_trick.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print("Found contours with RETR_LIST:", len(contours))
    
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w)/h
        
        # Print info for all contours with area between 500 and 8000
        if 500 < area < 8000:
            print(f"Contour {i}: area={area}, bbox=({x},{y},{w},{h}), aspect_ratio={aspect_ratio:.2f}")
            crop = img[y:y+h, x:x+w]
            cv2.imwrite(f"/Users/admin/.gemini/antigravity-ide/brain/99eec12b-56cc-4eb2-b9c6-920fc66c5416/contour_crop_{i}.png", crop)

if __name__ == "__main__":
    main()
