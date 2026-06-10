#!/usr/bin/env python3
"""
Unit tests for BridgeAnalyzer (standardize_bid, clean_header_text).
"""

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
    print("\nALL analyzer unit tests passed!")


if __name__ == "__main__":
    main()
