import cv2
import numpy as np

def main():
    img = cv2.imread("debug_captures/1_ui_full.png")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Yellow/Orange HSV range
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print("Detected yellow/orange contours:")
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(c)
            print(f"Contour {i}: x={x}, y={y}, w={w}, h={h}, area={area}")
            # Save crop
            crop = img[y:y+h, x:x+w]
            cv2.imwrite(f"/Users/admin/.gemini/antigravity-ide/brain/99eec12b-56cc-4eb2-b9c6-920fc66c5416/yellow_box_{i}.png", crop)

if __name__ == "__main__":
    main()
