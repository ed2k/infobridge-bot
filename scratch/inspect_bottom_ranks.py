import cv2
import numpy as np
import os

def main():
    img = cv2.imread("debug/player_hand_area.png")
    if img is None:
        print("❌ Failed to load image")
        return
        
    h_strip = img.shape[0]
    w_strip = img.shape[1]
    print(f"Image shape: {img.shape}")
    
    # Let's see the vertical extent of white/grey pixels
    hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    card_mask = (hsv_full[:, :, 1] < 50) & (hsv_full[:, :, 2] > 170)
    row_sums = np.sum(card_mask, axis=1)
    print("Vertical card row sums:")
    for y, s in enumerate(row_sums):
        if s > 0:
            print(f"  Row {y:2d}: {s}")
            
    # Resizing
    scale = 60.0 / h_strip
    hand_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    # Peak detection
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
    
    peaks = []
    min_dist = 15
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

    print(f"Peaks found: {peaks}")
    
    os.makedirs("debug_test_crops", exist_ok=True)
    for idx, peak in enumerate(peaks):
        # Let's save three different crops for inspection
        for offset in [15, 22]:
            x_card = max(0, peak - offset)
            crop = hand_img[0:60, x_card:min(x_card + 40, hand_img.shape[1])]
            cv2.imwrite(f"debug_test_crops/crop_p{idx}_offset{offset}.png", crop)
            
            # Let's print the top-left 10x10 gray values of the rank area to check contrast
            rank_crop = crop[2:20, 2:18]
            gray = cv2.cvtColor(rank_crop, cv2.COLOR_BGR2GRAY)
            # Find min and max intensity
            mn, mx, _, _ = cv2.minMaxLoc(gray)
            print(f"  Peak {idx:2d} offset {offset}: min={mn:.1f}, max={mx:.1f}, shape={gray.shape}")

if __name__ == "__main__":
    main()
