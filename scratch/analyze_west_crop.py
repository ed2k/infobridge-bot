import cv2
import pytesseract
import csv
from io import StringIO
import os

def main():
    img = cv2.imread("debug_captures/1_ui_full.png")
    if img is None:
        print("Error: debug_captures/1_ui_full.png not found")
        return
        
    x, y, w, h = 0, 350, 68, 430
    crop = img[y:y+h, x:x+w]
    
    os.makedirs("debug_test_crops", exist_ok=True)
    cv2.imwrite("debug_test_crops/west_crop_raw.png", crop)
    
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("debug_test_crops/west_crop_gray.png", gray)
    
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite("debug_test_crops/west_crop_binary.png", binary)
    
    print(f"Crop size: {crop.shape}")
    
    # Try different scales and configurations
    for fx in [1.0, 2.0, 3.0, 4.0]:
        scaled_gray = cv2.resize(gray, (0, 0), fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
        # Try a few thresh values
        for thresh_val in [None, 127, 180]:
            if thresh_val is not None:
                _, proc = cv2.threshold(scaled_gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
            else:
                proc = scaled_gray
                
            for psm in [11, 6, 12, 10]:
                for wl in [True, False]:
                    config = f"--psm {psm}"
                    if wl:
                        config += " -c tessedit_char_whitelist=AKQJT1098765432"
                    try:
                        data_str = pytesseract.image_to_data(proc, config=config, output_type=pytesseract.Output.STRING)
                        f = StringIO(data_str)
                        reader = csv.reader(f, delimiter='\t')
                        header = next(reader)
                        
                        text_idx = header.index('text')
                        conf_idx = header.index('conf')
                        left_idx = header.index('left')
                        top_idx = header.index('top')
                        
                        results = []
                        for row in reader:
                            if len(row) <= text_idx:
                                continue
                            text = row[text_idx].strip()
                            if not text:
                                continue
                            conf = float(row[conf_idx])
                            cx = int(int(row[left_idx]) / fx)
                            cy = int(int(row[top_idx]) / fx)
                            results.append((text, conf, cx, cy))
                            
                        if results:
                            # Print matching configurations
                            print(f"Scale={fx}, Thresh={thresh_val}, PSM={psm}, WL={wl}: found {len(results)} items")
                            for text, conf, cx, cy in results:
                                print(f"  {text!r:10} | Conf: {conf:5.1f}% | cx={cx}, cy={cy} (Global y={y+cy})")
                    except Exception as e:
                        print(f"Error on scale={fx}, Thresh={thresh_val}, PSM={psm}, WL={wl}: {e}")

if __name__ == "__main__":
    main()
