import cv2
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paddle_ocr import ocr_with_positions

def main():
    img_path = "debug/bidding_area.png"
    img = cv2.imread(img_path)
    print(f"Loaded image: {img_path}, shape: {img.shape}")
    
    # PaddleOCR expects BGR
    # Let's upscale like in the code
    fx = 4.0
    processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed = cv2.resize(processed, (0, 0), fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
    processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    
    detections = ocr_with_positions(processed_bgr, min_confidence=0.1)
    print("\n--- Raw PaddleOCR Detections (coordinates scaled back to original image space) ---")
    for text, cx, cy, w, h in detections:
        orig_cx = cx / fx
        orig_cy = cy / fx
        orig_w = w / fx
        orig_h = h / fx
        print(f"Text: {text:15s} | cx: {orig_cx:6.1f}, cy: {orig_cy:6.1f} | w: {orig_w:5.1f}, h: {orig_h:5.1f}")

if __name__ == "__main__":
    main()
