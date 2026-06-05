import cv2
import numpy as np
import os
import mss

def main():
    # Capture the entire screen width (e.g. 1920px) at the hand y-coordinate (871)
    # Let's capture a region of height 65, from x=0 to 1920, at y=850 (adjust globally to capture y=871)
    try:
        import pyautogui
        screen_w, screen_h = pyautogui.size()
    except Exception:
        screen_w, screen_h = 1920, 1080
        
    print(f"Screen size: {screen_w}x{screen_h}")
    
    # We capture from y = 850 to 930 (height 80), x = 0 to screen_w
    monitor = {
        "top": 850,
        "left": 0,
        "width": int(screen_w),
        "height": 80
    }
    
    with mss.mss() as sct:
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
    print(f"Captured screen slice (shape: {img_bgr.shape})")
    
    # Convert to HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # Red + black masks
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([25, 255, 255])
    lower_red2 = np.array([165, 40, 40])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    mask_black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 60, 100]))
    mask_suit = mask_red + mask_black
    
    # Look at y-slice corresponding to y=871 (which is index 21 relative to top=850)
    # Let's sum vertically from y_rel = 15 to 45 (which is y=865 to 895)
    profile = np.sum(mask_suit[15:45, :] > 0, axis=0).astype(np.float32)
    
    # Apply smoothing
    kernel = np.ones(5) / 5.0
    smoothed = np.convolve(profile, kernel, mode='same')
    
    # Find peaks across the entire screen width
    peaks = []
    min_dist = 6
    for x in range(min_dist, len(smoothed) - min_dist):
        val = smoothed[x]
        if val >= 2.0:
            is_max = True
            for dx in range(-min_dist, min_dist + 1):
                if smoothed[x + dx] > val:
                    is_max = False
                    break
            if is_max:
                if not peaks or (x - peaks[-1]) >= min_dist:
                    peaks.append(x)
                    
    print(f"\nFound {len(peaks)} card peaks across the entire screen width:")
    print(f"Peaks global x: {peaks}")
    
    # Check if there are peaks around our hand ROI: 1260 to 1678
    hand_peaks = [p for p in peaks if 1200 <= p <= 1700]
    print(f"Peaks in hand ROI region (1200-1700): {len(hand_peaks)} peaks at x: {hand_peaks}")
    
if __name__ == "__main__":
    main()
