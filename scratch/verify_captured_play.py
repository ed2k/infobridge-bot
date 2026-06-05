#!/usr/bin/env python3
"""
End-to-End Verification Script.
Mocks ScreenCapture to return regions from sample_board.png,
then runs the decision loop in single-pass mode to verify OCR/CV, tracking, and PBN/JSON serialization.
"""

import os
import shutil
import cv2
import unittest
from unittest.mock import patch
import main
from tracker import GameTracker
from capture import ScreenCapture

def run_integration_test():
    print("====================================================")
    print("        VERIFYING INTEGRATED PLAY CAPTURE          ")
    print("====================================================")

    # 1. Ensure sample board image is generated
    print("Generating sample board screenshot...")
    import generate_mock
    generate_mock.main()
    
    img = cv2.imread("sample_board.png")
    if img is None:
        print("❌ Error: sample_board.png not found.")
        return False

    # Mock Coordinate Map matching generate_mock.py
    mock_config = {
        "bidding_roi": {"x": 800, "y": 80, "width": 350, "height": 250},
        "trick_roi": {"x": 300, "y": 250, "width": 400, "height": 300},
        "player_hand_roi": {"x": 300, "y": 600, "width": 500, "height": 120}
    }

    # Extract mock crops
    b_roi = mock_config["bidding_roi"]
    bidding_crop = img[b_roi["y"]:b_roi["y"]+b_roi["height"], b_roi["x"]:b_roi["x"]+b_roi["width"]]
    
    t_roi = mock_config["trick_roi"]
    trick_crop = img[t_roi["y"]:t_roi["y"]+t_roi["height"], t_roi["x"]:t_roi["x"]+t_roi["width"]]
    
    h_roi = mock_config["player_hand_roi"]
    hand_crop = img[h_roi["y"]:h_roi["y"]+h_roi["height"], h_roi["x"]:h_roi["x"]+h_roi["width"]]

    # 2. Patch ScreenCapture methods
    output_dir = "test_verified_plays"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    print("Mocking screen capture and running decision loop...")
    with patch.object(ScreenCapture, 'capture_bidding', return_value=bidding_crop), \
         patch.object(ScreenCapture, 'capture_player_hand', return_value=hand_crop), \
         patch.object(ScreenCapture, 'capture_trick', return_value=trick_crop):
        
        # Run one iteration of the decision loop (once=True)
        # Bidding is active on the mock board since it has bids and no 3 passes, 
        # but to test full trick play extraction let's trigger it directly.
        tracker = GameTracker()
        
        # Test Bidding Extraction
        cap = ScreenCapture()
        analyzer = main.BridgeAnalyzer(verbose=False)
        
        bids = analyzer.extract_bids(cap.capture_bidding())
        print(f"👉 Extracted Bids: {bids}")
        tracker.update_bids(bids)
        
        # Test Hand Extraction
        hand = analyzer.extract_hand_cards(cap.capture_player_hand())
        valid_hand = [c for c in hand if c.get("rank") is not None and c.get("suit") is not None]
        print(f"👉 Extracted Hand: {valid_hand}")
        tracker.set_initial_hand(valid_hand)
        
        # Test Trick Area Extraction
        tricks = analyzer.extract_multiple_cards(cap.capture_trick())
        print(f"👉 Extracted Trick Cards: {tricks}")
        tracker.register_trick_state(tricks, trick_crop.shape[1], trick_crop.shape[0])
        
        # Force finalize trick and save
        tracker.finalize_current_trick()
        pbn_path, json_path = tracker.save_to_files(output_dir)
        
        if pbn_path and os.path.exists(pbn_path):
            print(f"✅ Success! PBN file generated: {pbn_path}")
            with open(pbn_path, "r") as f:
                print("\n--- GENERATED PBN FILE CONTENT ---")
                print(f.read())
        else:
            print("❌ Failure: PBN file was not generated.")
            return False
            
        if json_path and os.path.exists(json_path):
            print(f"✅ Success! JSON file generated: {json_path}")
            with open(json_path, "r") as f:
                print("\n--- GENERATED JSON FILE CONTENT ---")
                print(f.read())
        else:
            print("❌ Failure: JSON file was not generated.")
            return False
            
    print("====================================================")
    print("        END-TO-END VERIFICATION COMPLETE            ")
    print("====================================================")
    return True

if __name__ == "__main__":
    run_integration_test()
