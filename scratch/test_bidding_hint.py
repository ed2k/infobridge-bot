#!/usr/bin/env python3
"""
Test Script for Bidding Hint Automation.
Verifies that locate_ui_text_button correctly identifies level digits and the "NT" button
on the mock board image, and validates offset calculation correctness.
"""

import os
import cv2
import sys

# Add parent directory to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analyzer import BridgeAnalyzer

def test_bidding_hint_detection():
    print("====================================================")
    print("         TESTING BIDDING HINT DETECTION             ")
    print("====================================================")
    
    # 1. Load sample_board.png
    img_path = "sample_board.png"
    if not os.path.exists(img_path):
        print(f"❌ Error: {img_path} not found. Generate it first.")
        sys.exit(1)
        
    img = cv2.imread(img_path)
    print(f"Loaded {img_path} ({img.shape[1]}x{img.shape[0]})")
    
    # 2. Define mock ui_roi covering the entire image (so top-left is 0,0)
    ui_roi = {
        "x": 0,
        "y": 0,
        "width": img.shape[1],
        "height": img.shape[0]
    }
    
    analyzer = BridgeAnalyzer(verbose=True)
    
    # 3. Test Level Button Detection
    print("\n--- Testing Level Buttons '1'-'7' ---")
    for lvl in ["1", "2", "3", "4", "5", "6", "7"]:
        coords = analyzer.locate_ui_text_button(img, lvl, ui_roi, max_y=600)
        if coords:
            print(f"✅ Found level '{lvl}' at global coordinates: {coords}")
        else:
            print(f"❌ Failed to locate level button '{lvl}'")
            
    # 4. Test Special Button Detection
    print("\n--- Testing Special Buttons 'PASS' and 'DBL' ---")
    for spec in ["PASS", "DBL"]:
        coords = analyzer.locate_ui_text_button(img, spec, ui_roi, max_y=600)
        if coords:
            print(f"✅ Found special button '{spec}' at global coordinates: {coords}")
        else:
            print(f"❌ Failed to locate special button '{spec}'")
            
    # 5. Test NT Button Detection & Suit Coordinate Interpolation
    print("\n--- Testing NT Button & Suit Offset Interpolation ---")
    nt_coords = analyzer.locate_ui_text_button(img, "NT", ui_roi, max_y=600)
    if nt_coords:
        nx, ny = nt_coords
        print(f"✅ Found NT button at: {nt_coords}")
        
        # Test suit offsets
        suit_offsets = {
            "NT": 0,
            "S": -50,
            "H": -100,
            "D": -150,
            "C": -200
        }
        
        for suit, offset in suit_offsets.items():
            tx, ty = nx + offset, ny
            print(f"👉 Target coordinate for suit '{suit}' (offset={offset}): ({tx}, {ty})")
            
            # Verify coordinates are roughly aligned with the drawn suit buttons in generate_mock.py
            # Club target x: 850
            # Diamond target x: 900
            # Heart target x: 950
            # Spade target x: 1000
            # NT target x: 1050
            expected_x = 1050 + offset
            expected_y = 430
            
            # Check proximity (within 10 pixels vertically and horizontally)
            dx = abs(tx - expected_x)
            dy = abs(ty - expected_y)
            if dx <= 10 and dy <= 10:
                print(f"   ✅ Offset Coordinate matches expected center ({expected_x}, {expected_y}) within margin (dx={dx}, dy={dy})")
            else:
                print(f"   ❌ Offset Coordinate deviates from expected center ({expected_x}, {expected_y}): dx={dx}, dy={dy}")
    else:
        print("❌ Failed to locate NT button")
        
    print("\n====================================================")
    print("Test execution complete.")
    print("====================================================")

if __name__ == "__main__":
    test_bidding_hint_detection()
