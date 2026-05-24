#!/usr/bin/env python3
"""
Bridge Bot Main CLI Orchestrator.
Integrates calibration, capture, computer vision analysis, and mouse actions.
"""

import sys
import argparse
import time
import signal
import os
import cv2
import numpy as np
from capture import ScreenCapture
from analyzer import BridgeAnalyzer
from controller import BridgeController

def images_are_similar(img1, img2, threshold=1.0):
    """
    Checks if two images are visually similar using Mean Absolute Error (MAE).
    Very fast check to avoid redundant OCR/CV analysis on static screens.
    """
    if img1 is None or img2 is None:
        return False
    if img1.shape != img2.shape:
        return False
    mae = np.mean(cv2.absdiff(img1, img2))
    return mae < threshold

# Setup instant SIGINT handler to ensure prompt Control-C termination
def setup_signal_handler():
    def handle_sigint(sig, frame):
        print("\n👋 Terminating instantly...")
        os._exit(0)
    signal.signal(signal.SIGINT, handle_sigint)

setup_signal_handler()

def run_calibration():
    """Runs the calibration script."""
    print("🚀 Launching screen calibration...")
    import calibrate
    calibrate.main()

def run_capture_debug():
    """Captures all regions and saves them to disk."""
    print("📸 Capturing regions and saving debug images...")
    try:
        cap = ScreenCapture()
        cap.save_debug_images()
        print("✅ Captured successfully. Look in the 'debug_captures' folder to check boundaries.")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run calibration first: python main.py --calibrate")
    except Exception as e:
        print(f"❌ Error during capture: {e}")

def run_analysis(verbose=True):
    """Performs one-off screen capture and analysis."""
    print("🔍 Analyzing screen elements...")
    try:
        cap = ScreenCapture()
        analyzer = BridgeAnalyzer(verbose=verbose)
        
        # 1. Bids
        bidding_img = cap.capture_bidding()
        bids = analyzer.extract_bids(bidding_img)
        print("\n--- BIDS DETECTED ---")
        if bids:
            print(" -> ".join(bids))
        else:
            print("No bids detected (or OCR returned empty text).")

        # 2. Trick Area (played cards)
        trick_img = cap.capture_trick()
        # Draw some templates if needed, or run standard shape class
        detected_trick = analyzer.extract_multiple_cards(trick_img)
        print("\n--- TRICK AREA CARDS ---")
        if detected_trick:
            for i, card in enumerate(detected_trick):
                print(f" Card {i+1}: Rank={card['rank'] or '?'}, Suit={card['suit'] or '?'}")
        else:
            print("No played cards detected in the trick area.")

        # 3. Player's Hand
        hand_img = cap.capture_player_hand()
        detected_hand = analyzer.extract_hand_cards(hand_img)
        print("\n--- PLAYER HAND CARDS ---")
        if detected_hand:
            cards_str = [f"{c['rank'] or '?'}{c['suit'] or '?'}" for c in detected_hand]
            print(", ".join(cards_str))
        else:
            print("No cards detected in player's hand.")
            
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run calibration first: python main.py --calibrate")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")

def run_click():
    """Executes the mouse automation sequence to click build info."""
    print("🖱️ Running click build info automation...")
    try:
        ctrl = BridgeController()
        ctrl.click_build_info(return_to_start=True)
        print("✅ Click action performed successfully!")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run calibration first: python main.py --calibrate")
    except Exception as e:
        print(f"❌ Click action failed: {e}")

def run_monitoring(interval=2.0, verbose=False):
    """Monitors the screen for changes in bids/cards and reports them."""
    print(f"👁️ Starting bridge play UI monitor (polling every {interval}s)...")
    print("Press Ctrl+C to stop.")
    
    try:
        cap = ScreenCapture()
        analyzer = BridgeAnalyzer(verbose=verbose)
        ctrl = BridgeController()
        
        last_bids = []
        prev_bidding_img = None
        prev_trick_img = None
        bids = []
        detected_trick = []
        
        while True:
            # Capture and extract bids
            bidding_img = cap.capture_bidding()
            if prev_bidding_img is not None and images_are_similar(prev_bidding_img, bidding_img, threshold=1.0):
                # Similar image, reuse bids
                pass
            else:
                bids = analyzer.extract_bids(bidding_img)
                prev_bidding_img = bidding_img.copy() if bidding_img is not None else None
            
            # Check for changes in bids
            if bids != last_bids:
                print(f"\n📢 Bids changed! {time.strftime('%H:%M:%S')}")
                prev_str = " -> ".join([f"{direction}:{bid}" for direction, bid in last_bids]) if last_bids else "None"
                curr_str = " -> ".join([f"{direction}:{bid}" for direction, bid in bids]) if bids else "None"
                print(f"Previous: {prev_str}")
                print(f"Current : {curr_str}")
                last_bids = bids
                
            # Perform trick check
            trick_img = cap.capture_trick()
            if prev_trick_img is not None and images_are_similar(prev_trick_img, trick_img, threshold=1.0):
                # Similar image, reuse detected_trick
                pass
            else:
                detected_trick = analyzer.extract_multiple_cards(trick_img)
                prev_trick_img = trick_img.copy() if trick_img is not None else None
                
            if detected_trick:
                # Log current trick state
                trick_str = ", ".join([f"{c['rank'] or '?'}{c['suit'] or '?'}" for c in detected_trick])
                # We overwrite the line or print on update
                sys.stdout.write(f"\rCurrent Trick: {trick_str} (Detected {len(detected_trick)} cards)   ")
                sys.stdout.flush()
                
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run calibration first: python main.py --calibrate")
    except Exception as e:
        print(f"❌ Monitoring encountered an error: {e}")

def format_compact_hand(hand):
    """Formats a list of cards into a standard bridge compact hand representation."""
    suits = {"spade": [], "heart": [], "diamond": [], "club": []}
    suit_symbols = {"spade": "♠", "heart": "♥", "diamond": "♦", "club": "♣"}
    
    for card in hand:
        rank = card.get("rank")
        suit = card.get("suit")
        if rank and suit in suits:
            r_str = "10" if rank == "10" or rank == "T" else rank
            suits[suit].append(r_str)
            
    # Sort ranks high to low
    rank_order = {r: idx for idx, r in enumerate(["A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2"])}
    
    parts = []
    # Standard order: Spades, Hearts, Diamonds, Clubs
    for suit in ["spade", "heart", "diamond", "club"]:
        ranks = sorted(suits[suit], key=lambda r: rank_order.get(r, 99))
        if ranks:
            parts.append(f"{suit_symbols[suit]}{''.join(ranks)}")
            
    return "  ".join(parts)

def format_compact_bids(bids):
    """Formats a list of bids into a compact string."""
    if not bids:
        return "None"
    compact_list = []
    for b in bids:
        b_clean = b.upper()
        if b_clean == "PASS":
            compact_list.append("P")
        elif b_clean == "DBL":
            compact_list.append("X")
        elif b_clean == "RDBL":
            compact_list.append("XX")
        else:
            compact_list.append(b)
    return "-".join(compact_list)

def format_bidding_table(direction_bids):
    """
    Formats the chronological list of direction-bid tuples into a table under N, E, S, W columns.
    direction_bids: list of tuples (direction, bid_text), e.g. [('E', 'PASS'), ('S', 'PASS')]
    """
    cols = ["N", "E", "S", "W"]
    rows = []
    current_row = ["-"] * 4
    
    for direction, bid in direction_bids:
        col_idx = cols.index(direction) if direction in cols else 0
        
        # Format the bid text compactly for the table (e.g. PASS -> P)
        b_clean = bid.upper()
        if b_clean == "PASS":
            b_compact = "P"
        elif b_clean == "DBL":
            b_compact = "X"
        elif b_clean == "RDBL":
            b_compact = "XX"
        else:
            b_compact = bid
            
        # Clockwise chronological check: if col_idx is before or equal to an already filled column index, start a new row.
        need_new_row = False
        for i in range(4):
            if current_row[i] != "-":
                if i >= col_idx:
                    need_new_row = True
                    break
                    
        if need_new_row:
            rows.append(current_row)
            current_row = ["-"] * 4
            
        current_row[col_idx] = b_compact
        
    if any(val != "-" for val in current_row):
        rows.append(current_row)
        
    lines = []
    lines.append("   N      E      S      W")
    lines.append("=========================")
    for r in rows:
        lines.append(f"  {r[0]:<4}   {r[1]:<4}   {r[2]:<4}   {r[3]:<4}")
    return "\n".join(lines)

def run_decision_loop(interval=2.0, dry_run=False, verbose=False, once=False):
    """Runs the capture -> analyze -> decide -> execute loop continuously or once."""
    mode = "single pass" if once else "continuous"
    print(f"🤖 Starting Bridge Play Decision Engine ({mode}, polling every {interval}s, dry_run={dry_run})...")
    if not once:
        print("Press Ctrl+C to stop.")
    
    try:
        from strategy import decide_bid, decide_play_card
        cap = ScreenCapture()
        analyzer = BridgeAnalyzer(verbose=verbose)
        ctrl = BridgeController(dry_run=dry_run)
        
        last_bids = []
        last_trick_cards = []
        played_in_current_trick = False
        last_hand_len = 0
        last_action_time = 0
        COOLDOWN = 4.0  # Cooldown in seconds between actions to allow UI animation
        
        prev_bidding_img = None
        prev_hand_img = None
        prev_trick_img = None
        
        bids = []
        hand = []
        valid_hand = []
        trick_cards = []
        
        while True:
            current_time = time.time()
            
            # 1. Capture Bids
            bidding_img = cap.capture_bidding()
            if prev_bidding_img is not None and images_are_similar(prev_bidding_img, bidding_img, threshold=1.0):
                # Similar image, reuse bids
                pass
            else:
                bids = analyzer.extract_bids(bidding_img)
                prev_bidding_img = bidding_img.copy() if bidding_img is not None else None
            
            # 2. Capture Hand
            hand_img = cap.capture_player_hand()
            if prev_hand_img is not None and images_are_similar(prev_hand_img, hand_img, threshold=1.0):
                # Similar image, reuse hand and valid_hand
                pass
            else:
                hand = analyzer.extract_hand_cards(hand_img)
                valid_hand = [c for c in hand if c.get("rank") is not None and c.get("suit") is not None]
                prev_hand_img = hand_img.copy() if hand_img is not None else None
            
            # 3. Determine Game State
            # Bidding is active if there are bids and we haven't reached 3 consecutive passes
            flat_bids = [b[1] for b in bids]
            is_bidding_active = False
            if flat_bids:
                consecutive_passes = 0
                for b in reversed(flat_bids):
                    if b == "PASS":
                        consecutive_passes += 1
                    else:
                        break
                is_bidding_active = (consecutive_passes < 3) and len(flat_bids) < 40
            
            # If we have no cards left, the board is finished
            if not valid_hand:
                sys.stdout.write("\rBoard inactive or hand empty. Waiting...               ")
                sys.stdout.flush()
                # Reset prev images to force re-evaluation when cards reappear
                prev_hand_img = None
                prev_trick_img = None
                prev_bidding_img = None
                if once:
                    print("\nSingle-pass mode complete (no active hand detected).")
                    break
                time.sleep(interval)
                continue
                
            if is_bidding_active:
                # --- BIDDING STATE ---
                hand_str = format_compact_hand(valid_hand)
                bids_str = format_compact_bids(flat_bids)
                sys.stdout.write(f"\r[Bidding] Bids: {bids_str} | Hand: {hand_str}   ")
                sys.stdout.flush()
                
                if bids != last_bids:
                    last_bids = bids
                    table_str = format_bidding_table(bids)
                    suggested_bid = decide_bid(valid_hand, flat_bids)
                    print(f"\n📢 Bids updated:\n{table_str}")
                    print(f"👉 Suggested bid: {suggested_bid}")
            else:
                # --- PLAY STATE ---
                # Capture trick cards
                trick_img = cap.capture_trick()
                if prev_trick_img is not None and images_are_similar(prev_trick_img, trick_img, threshold=1.0):
                    # Similar image, reuse trick_cards
                    pass
                else:
                    trick_cards = analyzer.extract_multiple_cards(trick_img)
                    prev_trick_img = trick_img.copy() if trick_img is not None else None
                
                # If trick is cleared, reset played flag
                if len(trick_cards) == 0:
                    played_in_current_trick = False
                    
                # If trick cards changed, reset played flag in case a click was ignored
                if trick_cards != last_trick_cards:
                    last_trick_cards = trick_cards
                    # If someone else played, we might be allowed to play now
                    if played_in_current_trick and current_time - last_action_time > 1.5:
                        # If hand size did not decrease, the click was ignored
                        if len(valid_hand) == last_hand_len:
                            played_in_current_trick = False
                
                suit_symbols = {"spade": "♠", "heart": "♥", "diamond": "♦", "club": "♣"}
                trick_str = " ".join([f"{c['rank']}{suit_symbols.get(c['suit'], '')}" for c in trick_cards])
                hand_str = format_compact_hand(valid_hand)
                sys.stdout.write(f"\r[Play] Trick: [{trick_str}] | Hand: {hand_str}   ")
                sys.stdout.flush()
                
                if not played_in_current_trick:
                    # Decide card to play
                    card_idx, rationale = decide_play_card(hand, trick_cards)
                    if card_idx is not None and card_idx < len(hand):
                        chosen_card = hand[card_idx]
                        
                        # Only click if cooldown passed
                        if current_time - last_action_time > COOLDOWN:
                            print(f"\n🧠 Decision: {rationale}")
                            print(f"🎬 Action: Playing {chosen_card['rank']}{chosen_card['suit']}")
                            
                            # Execute play card action
                            ctrl.play_card(chosen_card["bbox"])
                            last_action_time = current_time
                            last_hand_len = len(valid_hand)
                            played_in_current_trick = True
                            
                            # Reset prev_hand_img and prev_trick_img to force immediate detection on next frames
                            prev_hand_img = None
                            prev_trick_img = None
                            
            if once:
                print("\nSingle-pass mode complete.")
                break

            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nDecision Loop stopped by user.")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run calibration first: python main.py --calibrate")
    except Exception as e:
        print(f"❌ Running loop failed: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Bridge Play UI Scraper & Automation Bot",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--calibrate", action="store_true", help="Run interactive screen calibration utility")
    group.add_argument("--capture-debug", action="store_true", help="Capture UI regions and save to debug_captures/")
    group.add_argument("--analyze", action="store_true", help="Run OCR/CV analysis on the live screen once")
    group.add_argument("--click", action="store_true", help="Click the 'Build Info' button using mouse automation")
    group.add_argument("--monitor", action="store_true", help="Start continuous bridge UI state monitor")
    group.add_argument("--run", action="store_true", help="Start continuous bridge play & decision runner")
    
    parser.add_argument("--interval", type=float, default=2.0, help="Polling interval for monitoring in seconds (default: 2.0)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (logs decisions and actions without moving mouse)")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode (logs detailed computer vision and template matching details)")
    parser.add_argument("--once", action="store_true", help="Run one iteration and exit (useful with --run)")

    args = parser.parse_args()

    if args.calibrate:
        run_calibration()
    elif args.capture_debug:
        run_capture_debug()
    elif args.analyze:
        run_analysis(verbose=args.verbose)
    elif args.click:
        run_click()
    elif args.monitor:
        run_monitoring(args.interval, verbose=args.verbose)
    elif args.run:
        run_decision_loop(args.interval, args.dry_run, verbose=args.verbose, once=args.once)

if __name__ == "__main__":
    main()
