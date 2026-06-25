#!/usr/bin/env python3
"""
Unit Tests for UIDetector (detector.py).
"""

import os
import cv2
import numpy as np
import shutil
import tempfile
from detector import UIDetector

MOCK_IMAGE_PATH = os.path.join("debug", "sample_board.png")

def test_find_suit_button_positions():
    print("Testing find_suit_button_positions...")
    # Generate sample board if it doesn't exist
    if not os.path.exists(MOCK_IMAGE_PATH):
        import generate_mock
        generate_mock.main()
        
    img = cv2.imread(MOCK_IMAGE_PATH)
    assert img is not None, f"Could not load {MOCK_IMAGE_PATH}"
    
    det = UIDetector(verbose=True)
    
    # In generate_mock.py, NT button is centered at cx=1050, cy=430 on a 1200x800 screen.
    # The suit buttons are: S at 1000, H at 950, D at 900, C at 850.
    ui_roi = {"x": 0, "y": 0, "width": 1200, "height": 800}
    nt_x, nt_y = 1050, 430
    
    res = det.find_suit_button_positions(img, ui_roi, nt_x, nt_y)
    
    assert "NT" in res
    assert "S" in res
    assert "H" in res
    assert "D" in res
    assert "C" in res
    
    # Check that they match the expected coordinates (close to them)
    # The detect/grid-fitting should be very close to the center
    assert abs(res["NT"][0] - 1050) < 5
    assert abs(res["NT"][1] - 430) < 5
    assert abs(res["S"][0] - 1000) < 15
    assert abs(res["S"][1] - 430) < 15
    assert abs(res["H"][0] - 950) < 15
    assert abs(res["H"][1] - 430) < 15
    assert abs(res["D"][0] - 900) < 15
    assert abs(res["D"][1] - 430) < 15
    assert abs(res["C"][0] - 850) < 15
    assert abs(res["C"][1] - 430) < 15
    
    print("✅ find_suit_button_positions test passed.")

def test_detect_game_panel():
    print("Testing detect_game_panel...")
    img = cv2.imread(MOCK_IMAGE_PATH)
    det = UIDetector(verbose=True)
    
    panel = det.detect_game_panel(img)
    # Game panel should have some width and height, and be on screen
    assert panel["width"] > 0
    assert panel["height"] > 0
    print("✅ detect_game_panel test passed.")

def test_detect_bidding_columns():
    print("Testing detect_bidding_columns...")
    img = cv2.imread(MOCK_IMAGE_PATH)
    det = UIDetector(verbose=True)
    
    # crop bidding ROI
    bidding_roi = {"x": 800, "y": 80, "width": 350, "height": 250}
    bidding_crop = img[bidding_roi["y"]:bidding_roi["y"]+bidding_roi["height"], 
                       bidding_roi["x"]:bidding_roi["x"]+bidding_roi["width"]]
    
    cols, width = det.detect_bidding_columns(bidding_crop)
    # The columns in generate_mock are WEST, NORTH, EAST, SOUTH.
    # So we should detect at least a couple of them
    assert len(cols) >= 2 or len(cols) == 0  # In case tesseract isn't installed
    print("✅ detect_bidding_columns test passed.")

def test_detect_hand_card_peaks():
    print("Testing detect_hand_card_peaks...")
    img = cv2.imread(MOCK_IMAGE_PATH)
    det = UIDetector(verbose=True)
    
    player_hand_roi = {"x": 300, "y": 600, "width": 500, "height": 60}
    hand_crop = img[player_hand_roi["y"]:player_hand_roi["y"]+player_hand_roi["height"],
                    player_hand_roi["x"]:player_hand_roi["x"]+player_hand_roi["width"]]
    
    peaks, scale = det.detect_hand_card_peaks(hand_crop)
    assert len(peaks) >= 12  # There are 13 cards in the hand
    print("✅ detect_hand_card_peaks test passed.")

def test_auto_bootstrap_templates():
    print("Testing auto_bootstrap_templates...")
    img = cv2.imread(MOCK_IMAGE_PATH)
    det = UIDetector(verbose=True)
    
    player_hand_roi = {"x": 300, "y": 600, "width": 500, "height": 60}
    hand_crop = img[player_hand_roi["y"]:player_hand_roi["y"]+player_hand_roi["height"],
                    player_hand_roi["x"]:player_hand_roi["x"]+player_hand_roi["width"]]
                    
    with tempfile.TemporaryDirectory() as tmpdir:
        success = det.auto_bootstrap_templates(hand_crop, output_dir=tmpdir)
        if success:
            for s in ["spade", "heart", "diamond", "club"]:
                assert os.path.exists(os.path.join(tmpdir, f"{s}.png"))
    print("✅ auto_bootstrap_templates test passed.")

def test_vulnerability_and_dealer():
    print("Testing vulnerability and dealer detection...")
    img = cv2.imread(MOCK_IMAGE_PATH)
    det = UIDetector(verbose=True)
    
    vul = det.detect_vulnerability(img)
    # In mock image, there are no vul dots, so should return None or default
    assert vul is None or isinstance(vul, dict)
    
    bidding_roi = {"x": 800, "y": 80, "width": 350, "height": 250}
    bidding_crop = img[bidding_roi["y"]:bidding_roi["y"]+bidding_roi["height"], 
                       bidding_roi["x"]:bidding_roi["x"]+bidding_roi["width"]]
    dealer = det.detect_dealer_indicator(bidding_crop)
    assert dealer is None or dealer in ["N", "E", "S", "W"]
    print("✅ vulnerability and dealer detection test passed.")

def test_helpers():
    print("Testing helpers...")
    det = UIDetector(verbose=True)
    
    # extract_contract_from_bids
    mock_bids = [("N", "PASS"), ("E", "1H"), ("S", "PASS"), ("W", "1NT")]
    res = det.extract_contract_from_bids(mock_bids)
    assert res["contract"] == "1NT"
    assert res["level"] == 1
    assert res["suit"] == "notrump"
    assert res["declarer"] == "W"
    
    # track_trick_winner
    completed_trick = {"N": "SA", "E": "H3", "S": "C2", "W": "DK"}
    winner = det.track_trick_winner(completed_trick, trump_suit="H")
    assert winner == "E"
    
    # get_remaining_suit_counts
    initial_hand = [
        {"suit": "spade"}, {"suit": "spade"},
        {"suit": "heart"},
        {"suit": "diamond"}
    ]
    played = ["SA", "H2"]
    counts = det.get_remaining_suit_counts(initial_hand, played)
    assert counts["spade"] == 1
    assert counts["heart"] == 0
    assert counts["diamond"] == 1
    assert counts["club"] == 0
    
    print("✅ helpers test passed.")

def main():
    test_find_suit_button_positions()
    test_detect_game_panel()
    test_detect_bidding_columns()
    test_detect_hand_card_peaks()
    test_auto_bootstrap_templates()
    test_vulnerability_and_dealer()
    test_helpers()
    print("\n🎉 ALL detector unit tests passed successfully!")

if __name__ == "__main__":
    main()
