import cv2
import numpy as np
import os

def main():
    img_path = "debug_captures/1_ui_full.png"
    if not os.path.exists(img_path):
        print("❌ UI image not found.")
        return
        
    img = cv2.imread(img_path)
    # Crop the hand area: relative y 651 to 716, x 0 to 435
    # Wait, the wide crop in our test was x from 1215 to 1670.
    # Relative to ui_x = 1245, this is x_rel = 1215 - 1245 = -30.
    # Since x_rel starts at -30, we should capture from the global image!
    # Let's crop from the global screen image if possible, but we don't have the global screen image saved.
    # Wait! We saved 1_ui_full.png which is a crop of ui_roi (x: 1245, y: 220, w: 435, h: 830).
    # So we cannot go left of x=1245.
    # But wait! In test_wide_hand_roi.py, we captured the crop directly from the screen!
    # Let's capture the crop directly from the screen again, draw lines at the peaks, and save it!
    import mss
    monitor = {
        "top": 871,
        "left": 1215,
        "width": 455,
        "height": 65
    }
    with mss.mss() as sct:
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        hand_crop = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
    # Draw vertical lines at all 15 peak positions
    peaks = [21, 55, 62, 93, 101, 132, 140, 175, 213, 253, 291, 330, 369, 407, 446]
    
    vis = hand_crop.copy()
    for idx, x in enumerate(peaks):
        # Draw red line for RED peaks, blue for BLACK peaks
        color = (0, 0, 255) if idx in [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14] else (255, 0, 0)
        cv2.line(vis, (x, 0), (x, 65), color, 1)
        cv2.putText(vis, str(idx+1), (x - 5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
    os.makedirs("debug_ocr_test", exist_ok=True)
    cv2.imwrite("debug_ocr_test/visualized_peaks.png", vis)
    print("Saved visualized peaks to debug_ocr_test/visualized_peaks.png")

if __name__ == "__main__":
    main()
