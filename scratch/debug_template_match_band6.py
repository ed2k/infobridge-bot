import cv2
import numpy as np
import os
from analyzer import BridgeAnalyzer

def main():
    img_path = "debug_captures/1_ui_full.png"
    if not os.path.exists(img_path):
        print("❌ UI image not found.")
        return
        
    img = cv2.imread(img_path)
    print(f"Analyzing {img_path} (shape: {img.shape})")
    
    analyzer = BridgeAnalyzer(verbose=True)
    
    # Crop Band 6: relative y 651 to 716, x 15 to 433
    hand_crop = img[651:716, 15:433]
    h_strip = hand_crop.shape[0]
    w_strip = hand_crop.shape[1]
    
    print(f"Hand crop shape: {hand_crop.shape}")
    
    # Scale hand image height to 60 to match suit template scaling
    scale = 1.0
    if h_strip != 60:
        scale = 60.0 / h_strip
        hand_crop = cv2.resize(hand_crop, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h_strip = 60
        w_strip = hand_crop.shape[1]
        
    print(f"Resized hand crop shape: {hand_crop.shape}")
    
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([25, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Trace template matching scores
    scores = []
    print("\n--- Sliding window match trace for Band 6 ---")
    for x in range(0, w_strip - 15):
        y_start = 37
        y_end = 50
        suit_crop = hand_crop[y_start:y_end, x:x+13]
        
        # Color check
        hsv = cv2.cvtColor(suit_crop, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        red_ratio = np.sum(red_mask > 0) / (suit_crop.shape[0] * suit_crop.shape[1])
        is_red = red_ratio > 0.015
        
        allowed_suits = ["heart", "diamond"] if is_red else ["spade", "club"]
        gray_crop = cv2.cvtColor(suit_crop, cv2.COLOR_BGR2GRAY)
        
        all_scores = {}
        for name, template in analyzer.suit_templates.items():
            if template is not None:
                res = cv2.matchTemplate(gray_crop, template, cv2.TM_CCOEFF_NORMED)
                all_scores[name] = float(res[0][0])
                
        best_allowed_suit = max(allowed_suits, key=lambda s: all_scores.get(s, -1.0))
        best_allowed_score = all_scores.get(best_allowed_suit, -1.0)
        
        scores.append((x, best_allowed_score, best_allowed_suit))
        
        if best_allowed_score > 0.40 or x % 10 == 0:
            print(f"x={x:3d} | is_red={is_red} (ratio={red_ratio:.3f}) | Best Allowed: {best_allowed_suit} ({best_allowed_score:.3f}) | All: " + 
                  ", ".join([f"{k}:{v:.3f}" for k, v in all_scores.items()]))
            
    # Find peaks with score > 0.60
    peaks = []
    min_distance = 13
    for idx in range(1, len(scores) - 1):
        x, score, suit = scores[idx]
        if score > 0.60:
            is_peak = True
            start_check = max(0, idx - min_distance)
            end_check = min(len(scores), idx + min_distance)
            for j in range(start_check, end_check):
                if scores[j][1] > score:
                    is_peak = False
                    break
            if is_peak:
                if not peaks or (x - peaks[-1]["x_suit"]) >= min_distance:
                    peaks.append({"x_suit": x, "score": score, "suit": suit})
                    
    print(f"\nDetected peaks with score > 0.60: {len(peaks)}")
    for p in peaks:
        print(f"  x={p['x_suit']}, score={p['score']:.3f}, suit={p['suit']}")
        
if __name__ == "__main__":
    main()
