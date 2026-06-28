#!/usr/bin/env python3
"""
Unit tests for BridgeAnalyzer (standardize_bid, clean_header_text).
"""

import os
import cv2
from analyzer import BridgeAnalyzer


def make_analyzer():
    return BridgeAnalyzer(verbose=False)


def test_clean_header_text_exact():
    a = make_analyzer()
    assert a.clean_header_text("SOUTH") == "S"
    assert a.clean_header_text("NORTH") == "N"
    assert a.clean_header_text("EAST") == "E"
    assert a.clean_header_text("WEST") == "W"
    assert a.clean_header_text("S") == "S"
    assert a.clean_header_text("N") == "N"
    assert a.clean_header_text("E") == "E"
    assert a.clean_header_text("W") == "W"


def test_clean_header_text_case_insensitive():
    a = make_analyzer()
    assert a.clean_header_text("south") == "S"
    assert a.clean_header_text("North") == "N"
    assert a.clean_header_text("EAST") == "E"
    assert a.clean_header_text("west") == "W"


def test_clean_header_text_split_words():
    a = make_analyzer()
    assert a.clean_header_text("SOU") == "S"
    assert a.clean_header_text("NOR") == "N"
    assert a.clean_header_text("EA") == "E"
    assert a.clean_header_text("WE") == "W"


def test_clean_header_text_noise():
    a = make_analyzer()
    assert a.clean_header_text("S.") == "S"
    assert a.clean_header_text("N:") == "N"
    assert a.clean_header_text("E123") == "E"
    assert a.clean_header_text("W!") == "W"
    assert a.clean_header_text("SOUTH.") == "S"


def test_clean_header_text_empty():
    a = make_analyzer()
    assert a.clean_header_text("") is None
    assert a.clean_header_text("123") is None
    assert a.clean_header_text("@#$") is None


def test_standardize_bid_pass():
    a = make_analyzer()
    assert a.standardize_bid("PASS") == "PASS"
    assert a.standardize_bid("PAS") == "PASS"
    assert a.standardize_bid("PA") == "PASS"
    assert a.standardize_bid("PASSED") == "PASS"
    assert a.standardize_bid("pass") == "PASS"


def test_standardize_bid_dbl():
    a = make_analyzer()
    assert a.standardize_bid("DBL") == "DBL"
    assert a.standardize_bid("DOUBLE") == "DBL"
    assert a.standardize_bid("X") == "DBL"
    assert a.standardize_bid("dbl") == "DBL"


def test_standardize_bid_rdbl():
    a = make_analyzer()
    assert a.standardize_bid("RDBL") == "RDBL"
    assert a.standardize_bid("REDOUBLE") == "RDBL"
    assert a.standardize_bid("XX") == "RDBL"


def test_standardize_bid_level_suit():
    a = make_analyzer()
    assert a.standardize_bid("1NT") == "1NT"
    assert a.standardize_bid("2S") == "2S"
    assert a.standardize_bid("3H") == "3H"
    assert a.standardize_bid("4D") == "4D"
    assert a.standardize_bid("5C") == "5C"
    assert a.standardize_bid("7NT") == "7NT"


def test_standardize_bid_spaces():
    a = make_analyzer()
    assert a.standardize_bid("1 NT") == "1NT"
    assert a.standardize_bid("3  NT") == "3NT"
    assert a.standardize_bid(" 1S ") == "1S"


def test_standardize_bid_ocr_digit_typos():
    a = make_analyzer()
    assert a.standardize_bid("INT") == "1NT"
    assert a.standardize_bid("LNT") == "1NT"
    assert a.standardize_bid("TNT") == "1NT"
    assert a.standardize_bid("!NT") == "1NT"
    assert a.standardize_bid("|NT") == "1NT"


def test_standardize_bid_full_suit_names():
    a = make_analyzer()
    assert a.standardize_bid("1SPADES") == "1S"
    assert a.standardize_bid("2HEARTS") == "2H"
    assert a.standardize_bid("3DIAMONDS") == "3D"
    assert a.standardize_bid("4CLUBS") == "4C"


def test_standardize_bid_symbols():
    a = make_analyzer()
    assert a.standardize_bid("1@") == "1S"
    assert a.standardize_bid("2&") == "2C"


def test_standardize_bid_invalid():
    a = make_analyzer()
    assert a.standardize_bid("foo") == "FOO"
    assert a.standardize_bid("1") == "1"
    assert a.standardize_bid("ABC") == "ABC"


def test_standardize_bid_empty():
    a = make_analyzer()
    assert a.standardize_bid("") == ""


def test_stop_at_question_mark():
    a = make_analyzer()
    mock_detections = [
        ("W", 39.0, 21.0, 20.0, 20.0),
        ("N", 95.0, 21.0, 20.0, 20.0),
        ("E", 153.0, 21.0, 20.0, 20.0),
        ("S", 209.0, 21.0, 20.0, 20.0),
        
        ("Pass", 39.0, 52.0, 40.0, 20.0),
        ("1H", 95.0, 52.0, 30.0, 20.0),
        ("Pass", 153.0, 52.0, 40.0, 20.0),
        ("1S", 209.0, 52.0, 30.0, 20.0),
        
        ("Pass", 39.0, 84.0, 40.0, 20.0),
        ("?", 95.0, 84.0, 15.0, 20.0),
    ]
    upscaled = []
    for text, cx, cy, w, h in mock_detections:
        upscaled.append((text, cx * 4.0, cy * 4.0, w * 4.0, h * 4.0))
        
    from unittest.mock import patch
    with patch("analyzer.paddle_ocr_positions", return_value=upscaled):
        with patch("analyzer.HAS_PADDLE", True):
            import numpy as np
            mock_img = np.zeros((255, 250, 3), dtype=np.uint8)
            bids = a.extract_bids(mock_img)
            expected = [
                ("W", "PASS"),
                ("N", "1H"),
                ("E", "PASS"),
                ("S", "1S"),
                ("W", "PASS"),
            ]
            assert bids == expected, f"Expected {expected}, got {bids}"


def test_correct_bid_sequence():
    a = make_analyzer()
    # Test correction of same-color suits (Spade vs Club, Heart vs Diamond) when illegal progression is detected
    input_bids = [("W", "1C"), ("N", "1C")]
    corrected = a.correct_bid_sequence(input_bids)
    assert corrected == [("W", "1C"), ("N", "1S")]
    
    input_bids_2 = [("W", "1H"), ("N", "1D")]
    corrected_2 = a.correct_bid_sequence(input_bids_2)
    assert corrected_2 == [("W", "1H"), ("N", "1D")]
    
    input_bids_3 = [("W", "1D"), ("N", "1D")]
    corrected_3 = a.correct_bid_sequence(input_bids_3)
    assert corrected_3 == [("W", "1D"), ("N", "1H")]


def test_extract_bids_bidding_area_png():
    print("Testing extract_bids on bidding_area.png (mocked)...")
    a = make_analyzer()
    
    mock_detections = [
        ("S", 38.9, 20.5, 17.8, 20.5),
        ("W", 95.5, 20.9, 23.0, 21.2),
        ("N", 152.4, 20.9, 20.2, 21.2),
        ("E", 209.4, 20.8, 16.2, 19.0),
        ("1", 150.9, 52.5, 38.2, 29.5),
        ("2", 201.0, 52.1, 25.0, 27.2),
        ("Pass 3", 54.6, 84.9, 89.2, 27.2),
        ("4", 144.8, 85.0, 25.5, 26.0),
        ("X", 208.5, 84.5, 29.0, 32.0),
        ("Pass Pass Pass", 96.4, 118.9, 171.8, 26.2),
        ("Your turn", 124.2, 148.5, 72.0, 20.0),
        ("Hide auction", 124.0, 173.4, 67.5, 14.2)
    ]
    upscaled = []
    for text, cx, cy, w, h in mock_detections:
        upscaled.append((text, cx * 4.0, cy * 4.0, w * 4.0, h * 4.0))
        
    from unittest.mock import patch
    
    def side_effect_classify(suit_crop, return_score=False):
        if not hasattr(side_effect_classify, "call_count"):
            side_effect_classify.call_count = 0
        call_idx = side_effect_classify.call_count
        side_effect_classify.call_count += 1
        
        mapping = {
            0: ("spade", 1.0),
            1: ("diamond", 1.0),
            2: ("diamond", 1.0),
            3: ("heart", 1.0)
        }
        suit, score = mapping.get(call_idx, (None, 0.0))
        if return_score:
            return suit, score
        return suit
        
    with patch("analyzer.paddle_ocr_positions", return_value=upscaled):
        with patch("analyzer.HAS_PADDLE", True):
            with patch.object(a, "classify_bid_suit", side_effect=side_effect_classify):
                import numpy as np
                mock_img = np.zeros((255, 250, 3), dtype=np.uint8)
                bids = a.extract_bids(mock_img)
                expected = [
                    ("N", "1S"),
                    ("E", "2D"),
                    ("S", "PASS"),
                    ("W", "3D"),
                    ("N", "4H"),
                    ("E", "DBL"),
                    ("S", "PASS"),
                    ("W", "PASS"),
                    ("N", "PASS")
                ]
                assert bids == expected, f"Expected {expected}, got {bids}"
    print("✅ test_extract_bids_bidding_area_png passed.")


def main():
    test_clean_header_text_exact()
    test_clean_header_text_case_insensitive()
    test_clean_header_text_split_words()
    test_clean_header_text_noise()
    test_clean_header_text_empty()
    test_standardize_bid_pass()
    test_standardize_bid_dbl()
    test_standardize_bid_rdbl()
    test_standardize_bid_level_suit()
    test_standardize_bid_spaces()
    test_standardize_bid_ocr_digit_typos()
    test_standardize_bid_full_suit_names()
    test_standardize_bid_symbols()
    test_standardize_bid_invalid()
    test_standardize_bid_empty()
    test_stop_at_question_mark()
    test_correct_bid_sequence()
    test_extract_bids_bidding_area_png()
    print("\nALL analyzer unit tests passed!")


if __name__ == "__main__":
    main()
