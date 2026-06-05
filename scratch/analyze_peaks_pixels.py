import cv2
import numpy as np
import os

def main():
    img_path = "debug_ocr_test/visualized_peaks.png"
    if not os.path.exists(img_path):
        # Let's crop directly from screen if not found
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
    else:
        # Load the visualized image and convert back to original (without lines) by reloading from screen or loading the clean crop
        # Let's just capture a fresh clean crop from the screen to be absolutely sure!
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
            
    h_strip = hand_crop.shape[0]
    scale = 60.0 / h_strip
    hand_crop = cv2.resize(hand_crop, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
    
    peaks = [21, 55, 62, 93, 101, 132, 140, 175, 213, 253, 291, 330, 369, 407, 446]
    
    print("\n--- Analyzing Rank Area Dark Pixels ---")
    for idx, x_val in enumerate(peaks):
        # Rank area: relative y 9..35, x from x_val-5 to x_val+10
        x_start = max(0, x_val - 5)
        x_end = min(hand_crop.shape[1], x_val + 10)
        rank_crop = gray[9:35, x_start:x_end]
        
        # Count dark pixels (gray < 200)
        dark_pixels = np.sum(rank_crop < 200)
        
        # Also print the crop row standard deviations to see if there is text texture
        std_val = np.std(rank_crop)
        
        print(f"Peak {idx+1:2d} at x={x_val:3d} | Dark pixels (gray<200)={dark_pixels} | Std Dev={std_val:.2f}")

if __name__ == "__main__":
    main()
