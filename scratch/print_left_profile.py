import cv2
import numpy as np

def main():
    img = cv2.imread("debug/player_hand_area.png")
    if img is None:
        print("❌ Failed to load image")
        return
        
    h_strip = img.shape[0]
    w_strip = img.shape[1]
    
    scale = 60.0 / h_strip
    hand_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    w_strip = hand_img.shape[1]
    
    hsv = cv2.cvtColor(hand_img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([25, 255, 255])
    lower_red2 = np.array([165, 40, 40])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    mask_black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 60, 100]))
    mask_suit = mask_red + mask_black
    profile = np.sum(mask_suit[41:54, :] > 0, axis=0).astype(np.float32)
    kernel = np.ones(13) / 13.0
    smoothed = np.convolve(profile, kernel, mode='same')
    
    print("Profile values at left edge:")
    for x in range(0, 60):
        print(f"  x={x:3d}: raw={profile[x]:.1f}, smoothed={smoothed[x]:.3f}")

if __name__ == "__main__":
    main()
