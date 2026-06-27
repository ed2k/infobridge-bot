#!/usr/bin/env python3
"""
Test Suite for GameTracker.
Verifies hand serialization, seat classification, trick transitions, and PBN/JSON output.
"""

import os
import shutil
import tempfile
import json
from tracker import GameTracker

def test_hand_serialization():
    print("Testing Hand Serialization...")
    tracker = GameTracker()
    
    # Mock hand: AK3 of spades, QJT5 of hearts, 98 of diamonds, 762 of clubs
    mock_hand = [
        {"rank": "A", "suit": "spade"},
        {"rank": "K", "suit": "spade"},
        {"rank": "3", "suit": "spade"},
        {"rank": "Q", "suit": "heart"},
        {"rank": "J", "suit": "heart"},
        {"rank": "T", "suit": "heart"},
        {"rank": "5", "suit": "heart"},
        {"rank": "9", "suit": "diamond"},
        {"rank": "8", "suit": "diamond"},
        {"rank": "7", "suit": "club"},
        {"rank": "6", "suit": "club"},
        {"rank": "2", "suit": "club"}
    ]
    
    tracker.set_initial_hand(mock_hand)
    pbn_hand = tracker.get_south_hand_pbn()
    expected = "AK3.QJT5.98.762"
    
    assert pbn_hand == expected, f"Expected {expected}, got {pbn_hand}"
    print("✅ Hand Serialization test passed.")

def test_seat_classification():
    print("Testing Seat Classification...")
    tracker = GameTracker()
    w, h = 400, 300
    
    # North card (centered horizontally, top vertically)
    c_n = tracker.classify_seat({"x": 170, "y": 25, "w": 55, "h": 80}, w, h)
    assert c_n == "N", f"Expected N, got {c_n}"
    
    # South card (centered horizontally, bottom vertically)
    c_s = tracker.classify_seat({"x": 170, "y": 195, "w": 55, "h": 80}, w, h)
    assert c_s == "S", f"Expected S, got {c_s}"
    
    # East card (right side horizontally, centered vertically)
    c_e = tracker.classify_seat({"x": 260, "y": 110, "w": 55, "h": 80}, w, h)
    assert c_e == "E", f"Expected E, got {c_e}"
    
    # West card (left side horizontally, centered vertically)
    c_w = tracker.classify_seat({"x": 80, "y": 110, "w": 55, "h": 80}, w, h)
    assert c_w == "W", f"Expected W, got {c_w}"
    
    print("✅ Seat Classification test passed.")

def test_trick_transitions():
    print("Testing Trick Transitions...")
    tracker = GameTracker()
    w, h = 400, 300
    
    # Poll 1: North plays Spade Ace
    t1 = [{"rank": "A", "suit": "spade", "bbox": {"x": 170, "y": 25, "w": 55, "h": 80}}]
    tracker.register_trick_state(t1, w, h)
    assert tracker.first_lead == "N", f"Expected first lead to be N, got {tracker.first_lead}"
    assert tracker.current_trick["N"] == "SA", f"Expected SA, got {tracker.current_trick['N']}"
    assert len(tracker.completed_tricks) == 0
    
    # Poll 2: North Spade Ace, East plays Heart King
    t2 = [
        {"rank": "A", "suit": "spade", "bbox": {"x": 170, "y": 25, "w": 55, "h": 80}},
        {"rank": "K", "suit": "heart", "bbox": {"x": 260, "y": 110, "w": 55, "h": 80}}
    ]
    tracker.register_trick_state(t2, w, h)
    assert tracker.current_trick["E"] == "HK", f"Expected HK, got {tracker.current_trick['E']}"
    assert len(tracker.completed_tricks) == 0
    
    # Poll 3: North Spade Ace, East Heart King, South plays Club 10, West plays Diamond Queen
    t3 = [
        {"rank": "A", "suit": "spade", "bbox": {"x": 170, "y": 25, "w": 55, "h": 80}},
        {"rank": "K", "suit": "heart", "bbox": {"x": 260, "y": 110, "w": 55, "h": 80}},
        {"rank": "T", "suit": "club", "bbox": {"x": 170, "y": 195, "w": 55, "h": 80}},
        {"rank": "Q", "suit": "diamond", "bbox": {"x": 80, "y": 110, "w": 55, "h": 80}}
    ]
    tracker.register_trick_state(t3, w, h)
    assert tracker.current_trick["S"] == "CT", f"Expected CT, got {tracker.current_trick['S']}"
    assert tracker.current_trick["W"] == "DQ", f"Expected DQ, got {tracker.current_trick['W']}"
    assert len(tracker.completed_tricks) == 0
    
    # Poll 4: Trick is cleared
    tracker.register_trick_state([], w, h)
    assert len(tracker.completed_tricks) == 1
    assert tracker.completed_tricks[0] == {"N": "SA", "E": "HK", "S": "CT", "W": "DQ"}
    assert tracker.current_trick == {"N": None, "E": None, "S": None, "W": None}
    
    # Poll 5: East leads next trick (Club 3)
    t4 = [{"rank": "3", "suit": "club", "bbox": {"x": 260, "y": 110, "w": 55, "h": 80}}]
    tracker.register_trick_state(t4, w, h)
    assert tracker.current_trick["E"] == "C3", f"Expected C3, got {tracker.current_trick['E']}"
    assert len(tracker.completed_tricks) == 1
    
    # Poll 6: East Club 3, South plays Club 9
    t5 = [
        {"rank": "3", "suit": "club", "bbox": {"x": 260, "y": 110, "w": 55, "h": 80}},
        {"rank": "9", "suit": "club", "bbox": {"x": 170, "y": 195, "w": 55, "h": 80}}
    ]
    tracker.register_trick_state(t5, w, h)
    assert tracker.current_trick["S"] == "C9", f"Expected C9, got {tracker.current_trick['S']}"
    
    # Poll 7: Next trick starts immediately without seeing empty transition (e.g. West plays Spade 2 in new trick)
    # This should auto-finalize Trick 2 and start Trick 3 with West playing Spade 2.
    t6 = [{"rank": "2", "suit": "spade", "bbox": {"x": 80, "y": 110, "w": 55, "h": 80}}]
    tracker.register_trick_state(t6, w, h)
    assert len(tracker.completed_tricks) == 2
    assert tracker.completed_tricks[1] == {"N": None, "E": "C3", "S": "C9", "W": None}
    assert tracker.current_trick == {"N": None, "E": None, "S": None, "W": "S2"}
    
    print("✅ Trick Transitions test passed.")

def test_file_io():
    print("Testing File IO and Formats...")
    tracker = GameTracker()
    tracker.update_bids([("N", "PASS"), ("E", "1H"), ("S", "PASS"), ("W", "1NT")])
    tracker.set_initial_hand([
        {"rank": "A", "suit": "spade"},
        {"rank": "K", "suit": "spade"},
        {"rank": "J", "suit": "heart"},
        {"rank": "Q", "suit": "club"},
    ])
    tracker.completed_tricks.append({"N": "SA", "E": "H3", "S": "C2", "W": "DK"})
    tracker.first_lead = "N"

    with tempfile.TemporaryDirectory() as tmpdir:
        pbn_path, json_path = tracker.save_to_files(tmpdir)
        
        assert os.path.exists(pbn_path)
        assert os.path.exists(json_path)
        
        # Verify JSON content
        with open(json_path, "r") as f:
            data = json.load(f)
            assert data["dealer"] == "N"
            assert data["first_lead"] == "N"
            assert len(data["bids"]) == 4
            assert data["completed_tricks"][0] == {"N": "SA", "E": "H3", "S": "C2", "W": "DK"}
            
        # Verify PBN content
        with open(pbn_path, "r") as f:
            pbn = f.read()
            assert '[Dealer "N"]' in pbn
            assert '[Auction "N"]' in pbn
            assert 'Pass 1H Pass 1NT' in pbn
            assert '[Play "N"]' in pbn
            assert 'SA H3 C2 DK' in pbn

    print("✅ File IO test passed.")


def test_gamestate_bids_persistence():
    print("Testing GameState Bids Persistence...")
    from main import GameState
    state = GameState()
    
    # Initial state
    assert state.bids == []
    
    # Valid initial update
    state.update_bids([("N", "1C"), ("E", "PASS")])
    assert state.bids == [("N", "1C"), ("E", "PASS")]
    
    # Shorter update should be rejected/ignored (not overwrite)
    state.update_bids([("N", "1C")])
    assert state.bids == [("N", "1C"), ("E", "PASS")]
    
    # Equal or longer update should be accepted
    state.update_bids([("N", "1C"), ("E", "PASS"), ("S", "1S")])
    assert state.bids == [("N", "1C"), ("E", "PASS"), ("S", "1S")]
    
    # reset() should clear the sequence for a new round
    state.reset()
    assert state.bids == []
    print("✅ GameState Bids Persistence test passed.")


def main():
    test_hand_serialization()
    test_seat_classification()
    test_trick_transitions()
    test_file_io()
    test_gamestate_bids_persistence()
    print("\n🎉 ALL GameTracker tests passed successfully!")


if __name__ == "__main__":
    main()
