import cv2
import pytesseract

def main():
    img = cv2.imread("debug/player_hand_area.png")
    if img is None:
        print("❌ Failed to load image")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    
    for thresh in [100, 120, 150, "otsu"]:
        if thresh == "otsu":
            th = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        else:
            th = cv2.threshold(scaled, thresh, 255, cv2.THRESH_BINARY)[1]
            
        for inv in [False, True]:
            proc = cv2.bitwise_not(th) if inv else th
            text = pytesseract.image_to_string(proc, config="--psm 11").strip()
            if text:
                print(f"Thresh={thresh}, Inv={inv}:")
                print(repr(text))
                print("-" * 40)

if __name__ == "__main__":
    main()
