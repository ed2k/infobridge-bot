#!/usr/bin/env python3
"""
Test Runner for Bridge Bot.
Crops the generated sample_board.png using mock coordinate mapping, 
and runs OCR and Card Detection pipelines to verify functionality.
"""

import os
import cv2
import json
from analyzer import BridgeAnalyzer

def main():
    print("====================================================")
    print("             BRIDGE BOT - PIPELINE TEST              ")
    print("====================================================")
    
    # 1. Generate the mockup image
    print("Generating sample board screenshot...")
    import generate_mock
    generate_mock.main()
        
    # Read the full mock image
    img = cv2.imread("sample_board.png")
    if img is None:
        print("❌ Error: Could not read sample_board.png.")
        return
        
    print("Loaded sample_board.png successfully.")
    
    # Mock Coordinate Map matching the dimensions in generate_mock.py
    mock_config = {
        "bidding_roi": {"x": 800, "y": 80, "width": 350, "height": 250},
        "trick_roi": {"x": 300, "y": 250, "width": 400, "height": 300},
        "player_hand_roi": {"x": 300, "y": 600, "width": 500, "height": 60}
    }
    
    # Instantiate analyzer
    analyzer = BridgeAnalyzer(verbose=True)
    
    # ----------------------------------------------------
    # Test Bidding History OCR
    # ----------------------------------------------------
    print("\n--- 1. Testing Bidding History OCR ---")
    b_roi = mock_config["bidding_roi"]
    bidding_crop = img[b_roi["y"]:b_roi["y"]+b_roi["height"], b_roi["x"]:b_roi["x"]+b_roi["width"]]
    
    cv2.makedirs = lambda d: os.makedirs(d, exist_ok=True)
    os.makedirs("debug_test_crops", exist_ok=True)
    cv2.imwrite("debug_test_crops/test_bidding.png", bidding_crop)
    print("Saved crop to debug_test_crops/test_bidding.png")
    
    try:
        bids_with_dirs = analyzer.extract_bids(bidding_crop)
        print("Extracting bids...")
        if bids_with_dirs:
            bids = [b[1] for b in bids_with_dirs]
            bids_str = " -> ".join([f"{direction}:{bid}" for direction, bid in bids_with_dirs])
            print(f"👉 Success! Extracted Bids: {bids_str}")
        else:
            bids = []
            print("⚠️ OCR completed, but no valid bids were recognized.")
            print("Note: Ensure Tesseract OCR is installed (`brew install tesseract`).")
    except Exception as e:
        bids = []
        print(f"❌ Error during bidding OCR: {e}")
        
    # ----------------------------------------------------
    # Test Trick Area Card Detection (CV & OCR)
    # ----------------------------------------------------
    print("\n--- 2. Testing Trick Area Card Extraction ---")
    t_roi = mock_config["trick_roi"]
    trick_crop = img[t_roi["y"]:t_roi["y"]+t_roi["height"], t_roi["x"]:t_roi["x"]+t_roi["width"]]
    cv2.imwrite("debug_test_crops/test_trick.png", trick_crop)
    print("Saved crop to debug_test_crops/test_trick.png")
    
    try:
        detected_cards = analyzer.extract_multiple_cards(trick_crop)
        print("Extracting cards...")
        if detected_cards:
            print(f"👉 Success! Detected {len(detected_cards)} cards:")
            for idx, c in enumerate(detected_cards):
                print(f"   Card {idx+1}: Rank={c['rank']}, Suit={c['suit']}")
        else:
            print("⚠️ No cards detected in trick area. Check contour thresholding/filters.")
    except Exception as e:
        print(f"❌ Error during trick area card extraction: {e}")

    # ----------------------------------------------------
    # Test Player Hand Card Detection (CV & OCR)
    # ----------------------------------------------------
    print("\n--- 3. Testing Player Hand Card Extraction ---")
    h_roi = mock_config["player_hand_roi"]
    hand_crop = img[h_roi["y"]:h_roi["y"]+h_roi["height"], h_roi["x"]:h_roi["x"]+h_roi["width"]]
    cv2.imwrite("debug_test_crops/test_hand.png", hand_crop)
    print("Saved crop to debug_test_crops/test_hand.png")
    
    try:
        detected_hand = analyzer.extract_hand_cards(hand_crop)
        print("Extracting cards...")
        if detected_hand:
            cards_str = [f"{c['rank'] or '?'}{c['suit'] or '?'}" for c in detected_hand]
            print(f"👉 Success! Detected hand: {', '.join(cards_str)}")
            
            # ----------------------------------------------------
            # Test Strategy Decisions
            # ----------------------------------------------------
            print("\n--- 4. Testing Strategy Engine Decisions ---")
            from strategy import decide_bid, decide_play_card
            
            # Test Bidding Decision
            suggested_bid = decide_bid(detected_hand, bids)
            print(f"👉 Suggested Bid for hand (History: {bids}): {suggested_bid}")
            
            # Test Play Card Decision
            card_idx, rationale = decide_play_card(detected_hand, detected_cards)
            if card_idx is not None:
                chosen_card = detected_hand[card_idx]
                print(f"👉 Suggested Play Card: {chosen_card['rank']}{chosen_card['suit']}")
                print(f"   Rationale: {rationale}")
            else:
                print("⚠️ Failed to suggest card.")
                
        else:
            print("⚠️ No cards detected in player's hand.")
    except Exception as e:
        print(f"❌ Error during player hand card extraction: {e}")

    print("\n====================================================")
    print("Pipeline test complete.")
    print("====================================================")

if __name__ == "__main__":
    main()
