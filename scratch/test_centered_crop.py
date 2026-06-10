import cv2

def main():
    img = cv2.imread("debug_captures/3_trick.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w)/h
        
        if 1500 < area < 4000:
            card_crop = img[y:y+h, x:x+w]
            rank_crop = card_crop[2:int(h*0.46), int(w*0.10):int(w*0.90)]
            cv2.imwrite(f"/Users/admin/.gemini/antigravity-ide/brain/99eec12b-56cc-4eb2-b9c6-920fc66c5416/rank_crop_C{i}.png", rank_crop)
            print(f"Saved rank crop for Contour {i}")

if __name__ == "__main__":
    main()
