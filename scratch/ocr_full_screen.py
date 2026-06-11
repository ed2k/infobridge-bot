import cv2
import pytesseract

def main():
    img = cv2.imread("debug_captures/1_ui_full.png")
    if img is None:
        print("Failed to load image")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Scale up
    scaled = cv2.resize(gray, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    
    # Run OCR on the full screen
    txt = pytesseract.image_to_string(scaled)
    print("--- Full Screen OCR Output ---")
    print(txt)

if __name__ == "__main__":
    main()
