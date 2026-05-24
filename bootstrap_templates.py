#!/usr/bin/env python3
"""
Bootstrap Suit Templates.
Extracts spade, heart, club, and diamond templates from the user's actual screen crop
using mathematically precise horizontal offsets.
"""

import os
import cv2

def main():
    hand_path = "debug_captures/4_player_hand.png"
    if not os.path.exists(hand_path):
        print(f"❌ Error: {hand_path} not found. Please run 'python main.py --capture-debug' first.")
        return
        
    img = cv2.imread(hand_path)
    if img is None:
        print(f"❌ Error: Could not read {hand_path}.")
        return

    os.makedirs("templates", exist_ok=True)
    
    # Exact card positions determined from the user's 1680x1050 browser layout
    # These represent the exact starting coordinate of the 13x13 suit crop box
    targets = {
        "spade": 7,       # Card 0 (6 of Spades)
        "heart": 111,     # Card 4 (Ace of Hearts)
        "club": 215,      # Card 8 (Ace of Clubs)
        "diamond": 267    # Card 10 (Ace of Diamonds)
    }
    
    print("✂️ Extracting suit templates from your screen...")
    for suit, x_crop_start in targets.items():
        # Suit crop region: y from 37 to 50, x width is 13px (13x13 template)
        y_start = 37
        y_end = 50
        x_crop_end = x_crop_start + 13
        
        crop = img[y_start:y_end, x_crop_start:x_crop_end]
        
        # Save as grayscale template
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        out_path = os.path.join("templates", f"{suit}.png")
        cv2.imwrite(out_path, gray_crop)
        print(f"   Saved templates/{suit}.png (Crop: X={x_crop_start}..{x_crop_end})")
        
    print("\n🎉 Templates bootstrapped successfully! Running analysis now will use template matching.")

if __name__ == "__main__":
    main()
