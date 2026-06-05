import cv2
import sys
import os
from analyzer import BridgeAnalyzer

def main():
    img_path = "debug_captures/4_player_hand.png"
    if not os.path.exists(img_path):
        print(f"❌ Error: {img_path} does not exist.")
        return
        
    img = cv2.imread(img_path)
    print(f"Loaded {img_path} (shape: {img.shape})")
    
    analyzer = BridgeAnalyzer(verbose=True)
    detected_hand = analyzer.extract_hand_cards(img)
    
    print("\n--- DETECTED HAND ---")
    if detected_hand:
        cards_str = [f"{c['rank'] or '?'}{c['suit'] or '?'}" for c in detected_hand]
        print(f"Detected {len(detected_hand)} cards:")
        print(", ".join(cards_str))
    else:
        print("No cards detected.")

if __name__ == "__main__":
    main()
