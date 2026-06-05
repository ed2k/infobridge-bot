#!/usr/bin/env python3
"""
Test Suite for Bidding Explanation feature.
Generates mock bid screen, mocks click coordinates and screen grab, 
and validates end-to-end extraction of tooltip explanation.
"""

import os
import shutil
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from unittest.mock import patch
from analyzer import BridgeAnalyzer
from controller import BridgeController
from capture import ScreenCapture
import main

def get_font(size):
    for path in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc"
    ]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()

def run_explain_bid_test():
    print("====================================================")
    print("         TESTING BID EXPLANATION INTERACTION        ")
    print("====================================================")

    b_roi = {"x": 800, "y": 80, "width": 350, "height": 250}
    mock_config = {
        "bidding_roi": b_roi,
        "trick_roi": {"x": 300, "y": 250, "width": 400, "height": 300},
        "player_hand_roi": {"x": 300, "y": 600, "width": 500, "height": 120}
    }

    # Generate mock board
    import generate_mock
    generate_mock.main()
    
    img = cv2.imread("sample_board.png")
    if img is None:
        print("❌ Error: sample_board.png not found.")
        return False

    with patch.object(BridgeController, 'load_config', return_value=mock_config), \
         patch.object(ScreenCapture, 'load_config', return_value=mock_config):
         
        # 1. Test bbox extraction
        print("Testing bbox extraction from mock board...")
        bidding_crop = img[b_roi["y"]:b_roi["y"]+b_roi["height"], b_roi["x"]:b_roi["x"]+b_roi["width"]]

        analyzer = BridgeAnalyzer(verbose=False)
        bids = analyzer.extract_bids_with_bboxes(bidding_crop)
        
        if not bids:
            print("❌ Error: No bids detected with bboxes.")
            return False
            
        print(f"✅ Success! Extracted {len(bids)} bids with bounding boxes.")
        last_bid = bids[-1]
        print(f"   Last Bid: {last_bid['direction']}:{last_bid['bid']} at bbox {last_bid['bbox']}")

        # 2. Test click coordinate mapping (dry-run)
        print("\nTesting click coordinate mapping...")
        ctrl = BridgeController(dry_run=True)
        target_x, target_y = ctrl.click_bid(last_bid["bbox"])
        
        # Target coordinate should be bidding_roi x + bbox center x
        expected_x = int(b_roi["x"] + last_bid["bbox"]["x"] + (last_bid["bbox"]["w"] / 2))
        expected_y = int(b_roi["y"] + last_bid["bbox"]["y"] + (last_bid["bbox"]["h"] / 2))
        
        assert target_x == expected_x, f"Expected X: {expected_x}, Got: {target_x}"
        assert target_y == expected_y, f"Expected Y: {expected_y}, Got: {target_y}"
        print(f"✅ Success! Coordinate mapped correctly to ({target_x}, {target_y}).")

        # 3. Simulate and test end-to-end OCR popup extraction
        print("\nTesting popup explanation OCR extraction...")
        
        # Generate a mock tooltip popup image overlaying our sample board
        pil_img = Image.open("sample_board.png")
        draw = ImageDraw.Draw(pil_img)
        font = get_font(14)
        
        # Draw a mock tooltip bubble near the bid center coordinate
        tx_start = expected_x - 120
        ty_start = expected_y - 65
        draw.rounded_rectangle([tx_start, ty_start, tx_start+240, ty_start+45], radius=5, fill="#ffffcc", outline="black", width=1)
        draw.text((tx_start + 10, ty_start + 8), "Alert: 15-17 HCP Balanced", font=font, fill="black")
        draw.text((tx_start + 10, ty_start + 24), "Guarantees 2+ clubs", font=font, fill="black")
        
        simulated_screen = np.array(pil_img)
        simulated_screen_bgr = cv2.cvtColor(simulated_screen, cv2.COLOR_RGB2BGR)
        
        # Mock sct.grab to crop the simulated screen
        def mock_grab(monitor):
            top = monitor["top"]
            left = monitor["left"]
            w = monitor["width"]
            h = monitor["height"]
            crop = simulated_screen_bgr[top:top+h, left:left+w]
            crop_bgra = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
            class MockMSSGrab:
                def __init__(self, data):
                    self.data = data
                def __array__(self):
                    return self.data
            return MockMSSGrab(crop_bgra)

        # Patch ScreenCapture methods and click methods
        with patch.object(ScreenCapture, 'capture_bidding', return_value=bidding_crop), \
             patch.object(ScreenCapture, 'capture_player_hand', return_value=np.zeros((60, 500, 3), dtype=np.uint8)), \
             patch.object(ScreenCapture, 'capture_trick', return_value=np.zeros((300, 400, 3), dtype=np.uint8)), \
             patch('pyautogui.position', return_value=(0, 0)), \
             patch('pyautogui.moveTo'), \
             patch('pyautogui.click') as mock_click_py:
             
            # Replace sct.grab with our mock on the local screen capture object
            cap = ScreenCapture()
            cap.sct.grab = mock_grab
            
            with patch('main.ScreenCapture', return_value=cap), \
                 patch('main.BridgeController', return_value=ctrl):
                
                # Run stdout redirect to capture the printed explanation
                import sys
                from io import StringIO
                
                old_stdout = sys.stdout
                sys.stdout = mystdout = StringIO()
                
                try:
                    main.run_bid_explanation(verbose=False)
                finally:
                    sys.stdout = old_stdout
                    
                output = mystdout.getvalue()
                print("--- Captured main.run_bid_explanation output ---")
                print(output)
                print("------------------------------------------------")
                
                # Verify that the tooltip text was successfully extracted
                assert "Alert: 15-17 HCP Balanced" in output, "Failed to extract first line of tooltip"
                assert "Guarantees 2+ clubs" in output, "Failed to extract second line of tooltip"
                
    print("✅ Success! End-to-end popup OCR extraction test passed.")
    print("====================================================")
    print("         ALL INTERACTION TESTS COMPLETED            ")
    print("====================================================")
    return True

if __name__ == "__main__":
    if run_explain_bid_test():
        print("🎉 Verification tests passed!")
    else:
        print("❌ Verification tests failed.")
        exit(1)
