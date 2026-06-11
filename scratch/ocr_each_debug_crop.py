import cv2
import pytesseract
import os

def test_crop(suit_name):
    filepath = f"debug_ocr_test/dummy_suits/{suit_name}_crop.png"
    if not os.path.exists(filepath):
        print(f"Skipping {suit_name}: file does not exist")
        return
        
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    print(f"\n==================== Testing OCR on {suit_name}_crop.png (shape={img.shape}) ====================")
    for fx in [3.0, 4.0, 5.0]:
        for thresh_val in [127, 150, 180, "otsu"]:
            scaled = cv2.resize(gray, (0, 0), fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
            if thresh_val == "otsu":
                thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            else:
                thresh = cv2.threshold(scaled, thresh_val, 255, cv2.THRESH_BINARY)[1]
                
            for invert in [False, True]:
                proc = cv2.bitwise_not(thresh) if invert else thresh
                for psm in [6, 7, 8, 10, 11]:
                    try:
                        txt = pytesseract.image_to_string(proc, config=f"--psm {psm}")
                        txt_clean = txt.strip().replace("\n", " ")
                        if txt_clean:
                            print(f"fx={fx}, thresh={thresh_val}, invert={invert}, psm={psm}: '{txt_clean}'")
                    except Exception:
                        pass

if __name__ == "__main__":
    for suit in ["spade", "heart", "club", "diamond"]:
        test_crop(suit)
