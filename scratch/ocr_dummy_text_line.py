import cv2
import pytesseract
import os

def test_file(img_path):
    if not os.path.exists(img_path):
        print(f"Skipping {img_path}: does not exist")
        return
    img = cv2.imread(img_path)
    if img is None:
        print(f"Skipping {img_path}: failed to load")
        return
        
    print(f"\n--- Testing {img_path} ---")
    crop = img[275:310, 0:510]
    
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    for fx in [3.0, 4.0, 5.0]:
        for thresh_val in [127, 150, 180, "otsu"]:
            if thresh_val == "otsu":
                thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                # resize after thresholding is also an option, but let's do the same as main.py:
                scaled = cv2.resize(gray, (0, 0), fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
                thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            else:
                scaled = cv2.resize(gray, (0, 0), fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
                thresh = cv2.threshold(scaled, thresh_val, 255, cv2.THRESH_BINARY)[1]
                
            for invert in [False, True]:
                proc = cv2.bitwise_not(thresh) if invert else thresh
                for psm in [6, 7, 8, 11]:
                    try:
                        txt = pytesseract.image_to_string(proc, config=f"--psm {psm}")
                        txt_clean = txt.strip().replace("\n", " ")
                        if txt_clean:
                            print(f"fx={fx}, thresh={thresh_val}, invert={invert}, psm={psm}: '{txt_clean}'")
                    except Exception as e:
                        pass

def main():
    test_file("debug_captures/live_ui_all_sides.png")
    test_file("debug_captures/1_ui_full.png")

if __name__ == "__main__":
    main()
