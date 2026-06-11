import cv2
import os
from analyzer import BridgeAnalyzer
from main import detect_dummy_hands

def check_images():
    analyzer = BridgeAnalyzer(verbose=True)
    
    print("--- Checking Captured Images Info ---")
    files = {
        "1_ui_full.png": "debug_captures/1_ui_full.png",
        "2_bidding.png": "debug_captures/2_bidding.png",
        "3_trick.png": "debug_captures/3_trick.png",
        "4_player_hand.png": "debug_captures/4_player_hand.png",
        "5_bidding_hint.png": "debug_captures/5_bidding_hint.png"
    }
    
    loaded = {}
    for name, path in files.items():
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                print(f"File {name}: shape={img.shape}")
                loaded[name] = img
            else:
                print(f"File {name}: Failed to load!")
        else:
            print(f"File {name}: Does not exist!")
            
    # Test player hand
    if "4_player_hand.png" in loaded:
        print("\n--- Running Player Hand Detection ---")
        hand_img = loaded["4_player_hand.png"]
        try:
            detected_hand = analyzer.extract_hand_cards(hand_img)
            print("Detected hand cards:")
            for c in detected_hand:
                print(f"  Rank: {c.get('rank')}, Suit: {c.get('suit')}, Bbox: {c.get('bbox')}")
        except Exception as e:
            print(f"Error in extract_hand_cards: {e}")
            import traceback
            traceback.print_exc()

    # Test dummy hands
    if "1_ui_full.png" in loaded:
        print("\n--- Running Dummy Hand Detection ---")
        ui_img = loaded["1_ui_full.png"]
        try:
            dummies = detect_dummy_hands(ui_img, analyzer)
            for side, hand in dummies.items():
                print(f"  {side} Dummy Hand ({len(hand)} cards):")
                for c in hand:
                    print(f"    Rank: {c.get('rank')}, Suit: {c.get('suit')}, Bbox: {c.get('bbox')}")
        except Exception as e:
            print(f"Error in detect_dummy_hands: {e}")
            import traceback
            traceback.print_exc()

    # Test bidding
    if "2_bidding.png" in loaded:
        print("\n--- Running Bidding Detection ---")
        bidding_img = loaded["2_bidding.png"]
        try:
            bids = analyzer.extract_bids(bidding_img)
            print(f"Detected bids: {bids}")
        except Exception as e:
            print(f"Error in extract_bids: {e}")

if __name__ == "__main__":
    check_images()
