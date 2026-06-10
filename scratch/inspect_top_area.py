import cv2
import numpy as np

def main():
    img = cv2.imread("debug_captures/live_ui_all_sides.png")
    if img is None:
        print("Error: debug_captures/live_ui_all_sides.png not found")
        return
        
    h_img, w_img = img.shape[:2]
    print(f"Image size: {w_img}x{h_img}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Let's inspect all contours in y < 250
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print("\nAll contours in y < 250:")
    candidates = []
    for idx, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        if y < 250:
            print(f"Contour {idx}: x={x}..{x+w} (w={w}), y={y}..{y+h} (h={h}), area={cv2.contourArea(c)}")
            candidates.append((x, y, w, h))
            
    # Also let's print the average color/HSV of the regions to see if we can distinguish suits
    print("\nAnalyzing blocks:")
    candidates.sort()
    for idx, (x, y, w, h) in enumerate(candidates):
        crop = img[y:y+h, x:x+w]
        # Calculate red ratio
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([25, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        red_ratio = np.sum(red_mask > 0) / (w * h)
        
        # Calculate black/dark ratio (for black suits)
        # Background is white/light, suit icons are black or red.
        # Let's see if we can find suit icon bounding boxes inside the block.
        print(f"Block {idx}: x={x}..{x+w}, y={y}..{y+h}, red_ratio={red_ratio:.3f}")

if __name__ == "__main__":
    main()
