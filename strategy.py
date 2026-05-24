#!/usr/bin/env python3
"""
Bridge Strategy Engine.
Implements rule-based decision logic for bidding and card play.
"""

RANK_VALUES = {
    "A": 14, "K": 13, "Q": 12, "J": 11, "T": 10, "10": 10,
    "9": 9, "8": 8, "7": 7, "6": 6, "5": 5, "4": 4, "3": 3, "2": 2
}

HCP_VALUES = {
    "A": 4, "K": 3, "Q": 2, "J": 1
}

def calculate_hcp(hand):
    """Calculates High Card Points (HCP) for a hand of cards."""
    points = 0
    for card in hand:
        rank = card.get("rank")
        if rank in HCP_VALUES:
            points += HCP_VALUES[rank]
    return points

def get_suit_lengths(hand):
    """Returns a dict mapping each suit to the number of cards in that suit in hand."""
    lengths = {"spade": 0, "heart": 0, "diamond": 0, "club": 0}
    for card in hand:
        suit = card.get("suit")
        if suit in lengths:
            lengths[suit] += 1
    return lengths

def decide_bid(hand, bid_history):
    """
    Decides the next bid based on hand cards and bidding history.
    """
    hcp = calculate_hcp(hand)
    lengths = get_suit_lengths(hand)
    
    print(f"🤖 Strategy Bid Check: HCP={hcp}, Suit Counts={lengths}, History={bid_history}")
    
    # If we have no points, pass
    if hcp < 12:
        return "PASS"
        
    # Check if we can open 1NT (15-17 HCP, balanced hand)
    # Balanced hand: no voids, no singletons, at most one doubleton
    doubletons = sum(1 for count in lengths.values() if count == 2)
    singletons = sum(1 for count in lengths.values() if count <= 1)
    is_balanced = singletons == 0 and doubletons <= 1
    
    if 15 <= hcp <= 17 and is_balanced:
        # Check if 1NT is already bid or bypassed
        # Simple heuristic: if we can bid 1NT, do it, otherwise pass/bid suit
        return "1NT"
        
    # Open longest suit at 1 level
    longest_suit = max(lengths, key=lengths.get)
    suit_symbol = longest_suit[0].upper() # S, H, D, C
    
    # Decide appropriate level based on bids
    # In a simplified game, open at the 1 level or follow standard ladder
    bid_level = 1
    for b in reversed(bid_history):
        if b != "PASS" and b != "DBL" and b != "RDBL":
            # Extract level
            if b[0].isdigit():
                bid_level = max(bid_level, int(b[0]) + 1)
            break
            
    if bid_level > 7:
        return "PASS"
        
    return f"{bid_level}{suit_symbol}"

def decide_play_card(hand, trick_cards):
    """
    Decides which card to play from the hand based on the cards in the trick area.
    trick_cards: list of dicts like {'rank': 'A', 'suit': 'spade'} in play order.
    Returns the index of the chosen card in the hand list, and a rationale string.
    """
    if not hand:
        return None, "Empty hand"
        
    # Filter out invalid cards from hand
    valid_hand = [c for c in hand if c.get("rank") is not None and c.get("suit") is not None]
    if not valid_hand:
        # Fall back to first card in hand if all are invalid/unknown
        return 0, "No recognized card ranks in hand, playing first card"

    # If we are lead (first to play in the trick)
    if not trick_cards:
        # Heuristic: play the lowest card of our longest suit
        lengths = get_suit_lengths(valid_hand)
        longest_suit = max(lengths, key=lengths.get)
        
        suit_cards = [c for c in valid_hand if c["suit"] == longest_suit]
        # Find card with lowest rank value
        chosen_card = min(suit_cards, key=lambda c: RANK_VALUES.get(c["rank"], 0))
        
        # Find index in the original hand list
        orig_idx = hand.index(chosen_card)
        return orig_idx, f"Lead play: lowest card {chosen_card['rank']}{chosen_card['suit']} of longest suit {longest_suit}"

    # Determine led suit
    first_card = trick_cards[0]
    led_suit = first_card.get("suit")
    
    # Check if we can follow suit
    follow_cards = [c for c in valid_hand if c["suit"] == led_suit]
    
    if follow_cards:
        # Rule: follow suit!
        # Check current winning card in trick
        highest_trick_card = max(
            [c for c in trick_cards if c.get("suit") == led_suit],
            key=lambda c: RANK_VALUES.get(c["rank"], 0)
        )
        highest_val = RANK_VALUES.get(highest_trick_card["rank"], 0)
        
        # Can we beat it?
        winning_cards = [c for c in follow_cards if RANK_VALUES.get(c["rank"], 0) > highest_val]
        if winning_cards:
            # Play the smallest winning card to conserve higher cards
            chosen_card = min(winning_cards, key=lambda c: RANK_VALUES.get(c["rank"], 0))
            orig_idx = hand.index(chosen_card)
            return orig_idx, f"Follow suit: playing lowest winner {chosen_card['rank']}{chosen_card['suit']} to beat {highest_trick_card['rank']}"
        else:
            # Cannot beat it: play our lowest card to follow suit cheaply
            chosen_card = min(follow_cards, key=lambda c: RANK_VALUES.get(c["rank"], 0))
            orig_idx = hand.index(chosen_card)
            return orig_idx, f"Follow suit: playing lowest card {chosen_card['rank']}{chosen_card['suit']} since cannot beat trick"
            
    # Void in led suit: discard or trump
    # In simplified game, discard lowest card of shortest/weakest suit
    lengths = get_suit_lengths(valid_hand)
    shortest_suit = min([s for s, count in lengths.items() if count > 0], key=lengths.get)
    
    suit_cards = [c for c in valid_hand if c["suit"] == shortest_suit]
    chosen_card = min(suit_cards, key=lambda c: RANK_VALUES.get(c["rank"], 0))
    orig_idx = hand.index(chosen_card)
    return orig_idx, f"Discard: void in {led_suit}, discarding lowest {chosen_card['rank']}{chosen_card['suit']} from shortest suit {shortest_suit}"

if __name__ == "__main__":
    # Test strategy rules
    test_hand = [
        {"rank": "A", "suit": "spade"},
        {"rank": "K", "suit": "spade"},
        {"rank": "3", "suit": "heart"},
        {"rank": "Q", "suit": "club"}
    ]
    print("Test calculate_hcp:", calculate_hcp(test_hand))
    print("Test get_suit_lengths:", get_suit_lengths(test_hand))
    print("Test decide_bid:", decide_bid(test_hand, ["PASS", "1C"]))
    print("Test decide_play_card (lead):", decide_play_card(test_hand, []))
    print("Test decide_play_card (follow):", decide_play_card(test_hand, [{"rank": "T", "suit": "spade"}]))
