import cv2
import pytesseract

def main():
    img = cv2.imread("debug_captures/live_ui_all_sides.png")
    if img is None:
        print("Error")
        return
        
    # Crop the dummy text area
    crop = img[275:310, 0:510]
    
    # Let's save the crop so it is persisted for debugging
    cv2.imwrite("debug_captures/dummy_text_line.png", crop)
    
    # Run OCR with different PSM modes on grayscale, scaled up
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    for fx in [3.0, 4.0, 5.0]:
        scaled = cv2.resize(gray, (0, 0), fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
        # Try otsu and standard thresholding
        for thresh_val in [127, 150, 180, "otsu"]:
            if thresh_val == "otsu":
                thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            else:
                thresh = cv2.threshold(scaled, thresh_val, 255, cv2.THRESH_BINARY)[1]
                
            # Also try inverting (sometimes BBO uses dark on white or white on dark)
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

if __name__ == "__main__":
    main()
