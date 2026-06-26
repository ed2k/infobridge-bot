#!/usr/bin/env python3
"""
Game Tracker for Bridge Bot.
Tracks bidding history, player hands, and played tricks, and saves them to PBN/JSON files.
"""

import os
import json
import time
from datetime import datetime

class GameTracker:
    def __init__(self):
        self.bids = []  # List of (direction, bid_str)
        self.initial_hand = []  # List of card dicts: [{"rank": "A", "suit": "spade", "bbox": {...}}, ...]
        self.initial_dummy_hands = {"West": [], "North": [], "East": []}  # Remembered dummy hands
        self.completed_tricks = []  # List of dicts: {"N": card, "E": card, "S": card, "W": card}
        self.current_trick = {"N": None, "E": None, "S": None, "W": None}
        self.dealer = "N"  # Default dealer
        self.first_lead = None  # Seat that led the first trick of the play phase
        self.game_started_time = datetime.now()

    def update_bids(self, bids):
        """Updates the bidding history. Deduces dealer from the first bid."""
        if not bids:
            return
        # If the new list of bids is longer or different, update
        if len(bids) >= len(self.bids):
            self.bids = bids
            # Determine dealer (the seat of the first bid)
            if bids:
                self.dealer = bids[0][0]

    def set_initial_hand(self, hand):
        """Captures or refines the initial player hand at the start of play."""
        valid_cards = [c for c in hand if c.get("rank") is not None and c.get("suit") is not None]
        if not valid_cards:
            return
            
        played = self.get_played_cards_for_seat("S")
        if len(played) == 0:
            if len(valid_cards) > len(self.initial_hand):
                self.initial_hand = valid_cards

    def set_initial_dummy_hand(self, side, hand):
        """Captures or refines the initial dummy hand for a given side at the start of play."""
        valid_cards = [c for c in hand if c.get("rank") is not None and c.get("suit") is not None]
        if not valid_cards:
            return
            
        seat_map = {"West": "W", "North": "N", "East": "E"}
        seat = seat_map.get(side)
        if not seat:
            return
            
        played = self.get_played_cards_for_seat(seat)
        if len(played) == 0:
            if len(valid_cards) > len(self.initial_dummy_hands[side]):
                self.initial_dummy_hands[side] = valid_cards

    def get_played_cards_for_seat(self, seat):
        """Returns the list of PBN card notations played by a given seat in this game."""
        played = []
        for trick in self.completed_tricks:
            card = trick.get(seat)
            if card:
                played.append(card)
        card = self.current_trick.get(seat)
        if card:
            played.append(card)
        return played

    def get_all_played_cards(self):
        """Returns the set of all PBN card notations played by all seats in this game."""
        played = set()
        for trick in self.completed_tricks:
            for card in trick.values():
                if card:
                    played.add(card)
        for card in self.current_trick.values():
            if card:
                played.add(card)
        return played

    def register_trick_state(self, trick_cards, trick_width=400, trick_height=300):
        """
        Processes the current cards visible in the trick area.
        Detects trick completions and transitions.
        """
        poll_trick = {}
        for card in trick_cards:
            rank = card.get("rank")
            suit = card.get("suit")
            bbox = card.get("bbox")
            if rank and suit and bbox:
                seat = self.classify_seat(bbox, trick_width, trick_height)
                pbn_card = self.to_pbn_card(rank, suit)
                poll_trick[seat] = pbn_card

        if not poll_trick:
            # Trick area is empty. If we had cards in current_trick, finalize it!
            if any(val is not None for val in self.current_trick.values()):
                self.finalize_current_trick()
            return

        # Check for transitions:
        # 1. A seat already has a card in current_trick, but is playing again (conflict)
        # 2. The number of played cards decreased (meaning the trick cleared and a new one started)
        conflict = any(
            self.current_trick[seat] is not None and self.current_trick[seat] != card
            for seat, card in poll_trick.items()
        )
        count_decreased = sum(1 for v in poll_trick.values() if v is not None) < sum(1 for v in self.current_trick.values() if v is not None)

        if conflict or count_decreased:
            self.finalize_current_trick()

        # Update current trick state with the new card observations
        for seat, card in poll_trick.items():
            if self.current_trick[seat] is None:
                # If this is the very first card in the first trick, record it as the play phase leader
                if not self.first_lead and not self.completed_tricks and sum(1 for v in self.current_trick.values() if v is not None) == 0:
                    self.first_lead = seat
            self.current_trick[seat] = card

    def finalize_current_trick(self):
        """Finalizes the current trick and appends it to completed tricks."""
        if any(val is not None for val in self.current_trick.values()):
            # Record first lead if it wasn't recorded
            if not self.first_lead and not self.completed_tricks:
                for seat in ["N", "E", "S", "W"]:
                    if self.current_trick[seat] is not None:
                        self.first_lead = seat
                        break
            self.completed_tricks.append(self.current_trick.copy())
            self.current_trick = {"N": None, "E": None, "S": None, "W": None}

    def classify_seat(self, bbox, w, h):
        """Classifies a card's seat based on its bbox center relative to trick area center."""
        cx = bbox["x"] + bbox["w"] / 2
        cy = bbox["y"] + bbox["h"] / 2
        dx = cx - (w / 2)
        dy = cy - (h / 2)
        if abs(dx) > abs(dy):
            return "E" if dx > 0 else "W"
        else:
            return "S" if dy > 0 else "N"

    def to_pbn_card(self, rank, suit):
        """Formats rank and suit to PBN card notation (e.g. 'SA' for Spade Ace)."""
        r = rank.upper()
        if r == "10":
            r = "T"
        s_map = {"spade": "S", "heart": "H", "diamond": "D", "club": "C"}
        s = s_map.get(suit.lower(), "")
        return f"{s}{r}"

    def get_south_hand_pbn(self):
        """Formats South's starting hand into PBN hand notation (e.g. 'AK3.QJT5.98.762')."""
        if not self.initial_hand:
            return "?.?.?.?"
        
        suits = {"spade": [], "heart": [], "diamond": [], "club": []}
        for card in self.initial_hand:
            rank = card.get("rank")
            suit = card.get("suit")
            if rank and suit in suits:
                if rank not in suits[suit]:
                    suits[suit].append(rank)
                
        rank_order = {r: idx for idx, r in enumerate(["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"])}
        
        parts = []
        for suit in ["spade", "heart", "diamond", "club"]:
            sorted_ranks = sorted(suits[suit], key=lambda r: rank_order.get(r, 99))
            parts.append("".join(sorted_ranks) if sorted_ranks else "")
            
        return ".".join(parts)

    def generate_pbn(self):
        """Serializes the tracked game to PBN format string."""
        # Ensure any leftover trick is finalized
        self.finalize_current_trick()

        date_str = self.game_started_time.strftime("%Y.%m.%d")
        south_hand = self.get_south_hand_pbn()

        lines = [
            f'[Event "InfoBridge Live Capture"]',
            f'[Site "InfoBridge Bot"]',
            f'[Date "{date_str}"]',
            f'[Board "1"]',
            f'[West "?"]',
            f'[North "?"]',
            f'[East "?"]',
            f'[South "Bot"]',
            f'[Dealer "{self.dealer}"]',
            f'[Vulnerable "None"]',
            f'[Deal "N:? ? {south_hand} ?"]',
            ''
        ]

        # Format Bids/Auction
        if self.bids:
            lines.append(f'[Auction "{self.dealer}"]')
            
            def format_pbn_bid(bid):
                b_clean = bid.upper()
                if b_clean == "PASS":
                    return "Pass"
                elif b_clean == "DBL":
                    return "Dbl"
                elif b_clean == "RDBL":
                    return "Rdbl"
                return b_clean
                
            for i in range(0, len(self.bids), 4):
                chunk = self.bids[i:i+4]
                chunk_formatted = [format_pbn_bid(b[1]) for b in chunk]
                lines.append(" ".join(chunk_formatted))
            lines.append('')

        # Format Play
        if self.completed_tricks:
            leader = self.first_lead or "N"
            lines.append(f'[Play "{leader}"]')
            
            seats_order = ["N", "E", "S", "W"]
            leader_idx = seats_order.index(leader)
            play_order = seats_order[leader_idx:] + seats_order[:leader_idx]
            
            for trick in self.completed_tricks:
                trick_cards = []
                for seat in play_order:
                    card = trick.get(seat)
                    trick_cards.append(card if card else "-")
                lines.append(" ".join(trick_cards))
            lines.append('')

        return "\n".join(lines)

    def generate_json(self):
        """Serializes the tracked game to a dictionary/JSON structure."""
        # Ensure any leftover trick is finalized
        self.finalize_current_trick()

        return {
            "timestamp": self.game_started_time.isoformat(),
            "dealer": self.dealer,
            "first_lead": self.first_lead,
            "initial_hand": self.initial_hand,
            "bids": self.bids,
            "completed_tricks": self.completed_tricks
        }

    def save_to_files(self, output_dir="captured_plays"):
        """Saves both PBN and JSON outputs to files in output_dir."""
        if not self.bids and not self.initial_hand and not self.completed_tricks:
            # Nothing tracked to save
            return None, None

        os.makedirs(output_dir, exist_ok=True)
        file_prefix = f"play_{self.game_started_time.strftime('%Y%m%d_%H%M%S')}"
        
        pbn_path = os.path.join(output_dir, f"{file_prefix}.pbn")
        json_path = os.path.join(output_dir, f"{file_prefix}.json")

        # Save PBN
        pbn_content = self.generate_pbn()
        with open(pbn_path, "w") as f:
            f.write(pbn_content)

        # Save JSON
        json_data = self.generate_json()
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=4)

        return pbn_path, json_path
