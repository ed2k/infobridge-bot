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
from tracker import GameTracker
import pytesseract
import csv
from io import StringIO
from collections import Counter

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

def clean_rank_candidate(text):
    text = text.strip().upper()
    if not text:
        return None
    valid_ranks = {"A", "K", "Q", "J", "T", "10", "9", "8", "7", "6", "5", "4", "3", "2"}
    if text in valid_ranks:
        return text
    misreads = {
        "0": "Q", "O": "Q", "D": "Q",
        "1": "T",
        "J": "J", "L": "J", "I": "J",
    }
    if text in misreads:
        return misreads[text]
    return None

def detect_dummy_hands(img, analyzer):
    """
    Detects dummy hands on all sides (West, East, North) in the full UI capture image.
    Optimized for real-time execution in the main loop.
    """
    if img is None:
        return {"West": [], "North": [], "East": []}
        
    h_img, w_img = img.shape[:2]
    
    # 1. Compact dummy text line detection (y=275..310) - TRY FIRST!
    best_total = -1
    detected_cards = []
    
    if h_img >= 310:
        crop_w = w_img
        dummy_text_crop = img[275:310, 0:crop_w]
        
        # Determine suit order dynamically by checking colors in the 3rd and 4th groups
        # First, detect groups to find actual column positions
        hsv_temp = cv2.cvtColor(dummy_text_crop, cv2.COLOR_BGR2HSV)
        mask_red_temp = cv2.inRange(hsv_temp, np.array([0, 40, 40]), np.array([25, 255, 255])) + cv2.inRange(hsv_temp, np.array([165, 40, 40]), np.array([180, 255, 255]))
        gray_temp = cv2.cvtColor(dummy_text_crop, cv2.COLOR_BGR2GRAY)
        mask_total_temp = (gray_temp < 220)
        mask_black_temp = mask_total_temp & (mask_red_temp == 0)
        
        red_proj_temp = np.sum(mask_red_temp > 0, axis=0)
        black_proj_temp = np.sum(mask_black_temp > 0, axis=0)
        
        # Quick group detection to find column boundaries
        def quick_groups(active, min_size=30):
            groups = []
            in_group = False
            start = 0
            for x in range(len(active)):
                if active[x]:
                    if not in_group:
                        in_group = True
                        start = x
                else:
                    if in_group:
                        in_group = False
                        if groups and (start - groups[-1][1]) < 40:
                            groups[-1] = (groups[-1][0], x - 1)
                        else:
                            groups.append((start, x - 1))
            if in_group:
                if groups and (start - groups[-1][1]) < 40:
                    groups[-1] = (groups[-1][0], len(active) - 1)
                else:
                    groups.append((start, len(active) - 1))
            return [(s, e) for s, e in groups if (e - s + 1) >= min_size]
        
        red_active_temp = (red_proj_temp > 1)
        black_active_temp = (black_proj_temp > 1)
        all_quick_groups = sorted(quick_groups(red_active_temp) + quick_groups(black_active_temp))
        
        if len(all_quick_groups) >= 4:
            # Check colors in 3rd and 4th columns to determine suit order
            g3_x1, g3_x2 = all_quick_groups[2]
            g4_x1, g4_x2 = all_quick_groups[3]
            
            crop_col3 = dummy_text_crop[:, g3_x1:g3_x2]
            crop_col4 = dummy_text_crop[:, g4_x1:g4_x2]
            
            b3, g3, r3 = cv2.split(crop_col3)
            red3 = np.sum((r3.astype(int) > g3.astype(int) + 40) & (r3.astype(int) > b3.astype(int) + 40))
            
            b4, g4, r4 = cv2.split(crop_col4)
            red4 = np.sum((r4.astype(int) > g4.astype(int) + 40) & (r4.astype(int) > b4.astype(int) + 40))
            
            if red3 >= red4:
                suits_order = ["spade", "heart", "club", "diamond"]
            else:
                suits_order = ["spade", "heart", "diamond", "club"]
        else:
            suits_order = ["spade", "heart", "club", "diamond"]
            
        # Define expected colors
        expected_colors = []
        for suit in suits_order:
            if suit in ["spade", "club"]:
                expected_colors.append("BLACK")
            else:
                expected_colors.append("RED")
                
        # Define RED mask (Hearts/Diamonds)
        hsv = cv2.cvtColor(dummy_text_crop, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 40, 40])
        upper_red1 = np.array([25, 255, 255])
        lower_red2 = np.array([165, 40, 40])
        upper_red2 = np.array([180, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Define BLACK mask dynamically as any text pixel (gray < 220) that is not red
        gray_crop = cv2.cvtColor(dummy_text_crop, cv2.COLOR_BGR2GRAY)
        mask_total = (gray_crop < 220)
        mask_black = mask_total & (mask_red == 0)
        
        red_proj = np.sum(mask_red > 0, axis=0)
        black_proj = np.sum(mask_black > 0, axis=0)
        total_proj = red_proj + black_proj
        
        # Solid borders
        is_border = (total_proj >= 30)
        edge_mask = np.zeros_like(is_border)
        edge_mask[:5] = True
        edge_mask[crop_w - 5:] = True
        
        red_active = (red_proj > 1) & (~is_border) & (~edge_mask)
        black_active = (black_proj > 1) & (~is_border) & (~edge_mask)
        
        def get_groups(active_array, color_name):
            groups = []
            in_group = False
            start = 0
            for x in range(len(active_array)):
                if active_array[x]:
                    if not in_group:
                        in_group = True
                        start = x
                else:
                    if in_group:
                        in_group = False
                        if groups and (start - groups[-1][1]) < 40:
                            groups[-1] = (groups[-1][0], x - 1)
                        else:
                            groups.append((start, x - 1))
            if in_group:
                if groups and (start - groups[-1][1]) < 40:
                    groups[-1] = (groups[-1][0], len(active_array) - 1)
                else:
                    groups.append((start, len(active_array) - 1))
            return [(s, e, color_name) for s, e in groups if (e - s + 1) >= 5]
            
        red_groups = get_groups(red_active, "RED")
        black_groups = get_groups(black_active, "BLACK")
        
        all_groups = sorted(red_groups + black_groups)
        
        configs = [
            (4.0, "otsu", True, 6),
            (3.0, "otsu", True, 6),
        ]
        
        for fx_val, thresh_val, invert_val, psm_val in configs:
            current_detected = []
            
            # Match groups to suits by position
            # Sort groups by x position and assign suits sequentially
            sorted_groups = sorted(all_groups, key=lambda g: g[0])
            
            # Assign suits based on position using suits_order
            matched_suits = {suit: None for suit in suits_order}
            
            for i, suit_name in enumerate(suits_order):
                if i < len(sorted_groups):
                    matched_suits[suit_name] = (sorted_groups[i][0], sorted_groups[i][1])
                        
            for suit_name in suits_order:
                bounds = matched_suits[suit_name]
                if bounds is None:
                    continue
                    
                x1, x2 = bounds
                x1_pad = max(0, x1 - 3)
                x2_pad = min(crop_w, x2 + 3)
                
                col_crop = dummy_text_crop[:, x1_pad:x2_pad]
                col_gray = cv2.cvtColor(col_crop, cv2.COLOR_BGR2GRAY)
                
                scaled = cv2.resize(col_gray, (0, 0), fx=fx_val, fy=fx_val, interpolation=cv2.INTER_CUBIC)
                if thresh_val == "otsu":
                    thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                else:
                    thresh = cv2.threshold(scaled, thresh_val, 255, cv2.THRESH_BINARY)[1]
                    
                proc = cv2.bitwise_not(thresh) if invert_val else thresh
                
                try:
                    txt = pytesseract.image_to_string(proc, config=f"--psm {psm_val}")
                    
                    mapping = {
                        "0": "Q", "O": "Q", "D": "Q",
                        "S": "J", "N": "T", "W": "T",
                        "Z": "", "E": "6",
                        "M": "", "B": "", "I": "", "F": "", "H": "", "X": "",
                    }
                    cleaned = []
                    text_upper = txt.strip().upper().replace("\n", "").replace(" ", "").replace("10", "T")
                    for char in text_upper:
                        if char in mapping:
                            repl = mapping[char]
                            if repl:
                                cleaned.append(repl)
                        elif char.isdigit() or char in ["A", "K", "Q", "J", "T"]:
                            cleaned.append(char)
                            
                    filtered = []
                    for c in cleaned:
                        if not filtered or filtered[-1] != c:
                            filtered.append(c)
                            
                    rank_order = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
                    suit_cards = [r for r in filtered if r in rank_order]
                    
                    for r in suit_cards:
                        current_detected.append({
                            "rank": r,
                            "suit": suit_name,
                            "cx": int(x1 + (x2 - x1) / 2),
                            "cy": 290,
                            "bbox": {"x": x1, "y": 275, "w": x2 - x1, "h": 30}
                        })
                except Exception:
                    pass
            
            total_cards = len(current_detected)
            if 0 < total_cards <= 13:
                if total_cards > best_total:
                    best_total = total_cards
                    detected_cards = current_detected
                    if total_cards == 13:
                        break
                        
        if best_total > 0:
            north_dummy = detected_cards
            return {"West": [], "North": north_dummy, "East": []}


    # If compact detection failed, fall back to color profile peak detection (like player cards)
    # 2. Color Profile Peak Detection for North dummy cards (at the top)
    # This method uses the same approach as player card detection: find peaks in color profile
    contour_dummy_cards = []
    if h_img >= 100:
        # Crop the dummy area (top portion of UI)
        dummy_area = img[0:min(250, h_img), 0:w_img]
        
        # Normalize to height 60 like player cards
        target_h = 60
        scale = target_h / dummy_area.shape[0] if dummy_area.shape[0] != target_h else 1.0
        if scale != 1.0:
            dummy_area_norm = cv2.resize(dummy_area, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            dummy_area_norm = dummy_area.copy()
        
        h_norm, w_norm = dummy_area_norm.shape[:2]
        
        # HSV color masks (same as player card detection)
        hsv = cv2.cvtColor(dummy_area_norm, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 40, 40])
        upper_red1 = np.array([25, 255, 255])
        lower_red2 = np.array([165, 40, 40])
        upper_red2 = np.array([180, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
        mask_black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 60, 100]))
        mask_suit = mask_red + mask_black
        
        # 1D profile of suit row (y=41..54) - same as player cards
        suit_row_start = int(target_h * 0.68)
        suit_row_end = int(target_h * 0.90)
        profile = np.sum(mask_suit[suit_row_start:suit_row_end, :] > 0, axis=0).astype(np.float32)
        
        # Apply 1D smoothing (same kernel as player cards)
        kernel_size = max(7, int(w_norm * 0.015) | 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones(kernel_size) / float(kernel_size)
        smoothed = np.convolve(profile, kernel, mode='same')
        
        # Peak detection on smoothed profile
        peaks = []
        min_dist = 10
        for x in range(0, len(smoothed)):
            val = smoothed[x]
            if val >= 1.0:
                is_max = True
                for dx in range(-min_dist, min_dist + 1):
                    nx = x + dx
                    if 0 <= nx < len(smoothed) and smoothed[nx] > val:
                        is_max = False
                        break
                if is_max:
                    if not peaks or (x - peaks[-1]["x_suit"]) >= min_dist:
                        col_red = np.sum(mask_red[suit_row_start:suit_row_end, x] > 0)
                        col_black = np.sum(mask_black[suit_row_start:suit_row_end, x] > 0)
                        color = "RED" if col_red >= col_black else "BLACK"
                        peaks.append({"x_suit": x, "color": color})
        
        # Map x position to suit (North dummy: left to right = spade, heart, diamond/club)
        # Determine suit order dynamically by checking colors in columns 3 and 4
        crop_w = w_img
        col3_x1 = int(220 * (crop_w / 510.0))
        col3_x2 = int(330 * (crop_w / 510.0))
        col4_x1 = int(330 * (crop_w / 510.0))
        col4_x2 = int(480 * (crop_w / 510.0))
        
        # Check colors in columns 3 and 4 to determine suit order
        dummy_text_crop = img[275:310, 0:crop_w] if h_img >= 310 else dummy_area
        crop_col3 = dummy_text_crop[:, col3_x1:col3_x2]
        crop_col4 = dummy_text_crop[:, col4_x1:col4_x2]
        
        b3, g3, r3 = cv2.split(crop_col3)
        red3 = np.sum((r3.astype(int) > g3.astype(int) + 40) & (r3.astype(int) > b3.astype(int) + 40))
        
        b4, g4, r4 = cv2.split(crop_col4)
        red4 = np.sum((r4.astype(int) > g4.astype(int) + 40) & (r4.astype(int) > b4.astype(int) + 40))
        
        if red3 >= red4:
            suits_order = ["spade", "heart", "diamond", "club"]
        else:
            suits_order = ["spade", "heart", "club", "diamond"]
        
        # Map x position to suit based on peaks
        # Use equal-width columns for North dummy (spade, heart, diamond, club)
        col_width = w_img // 4
        suit_columns = {
            "spade": (0, col_width),
            "heart": (col_width, col_width * 2),
            "diamond": (col_width * 2, col_width * 3),
            "club": (col_width * 3, w_img)
        }
        
        # Reorder columns based on detected suit order
        suit_columns_ordered = [
            (suits_order[0], suit_columns["spade"]),
            (suits_order[1], suit_columns["heart"]),
            (suits_order[2], suit_columns[suits_order[2]]),
            (suits_order[3], suit_columns[suits_order[3]])
        ]
        
        for p in peaks:
            # Convert peak x to original image coordinates
            x_orig = int(p["x_suit"] / scale)
            
            # Determine suit from x position
            detected_suit = None
            for suit_name, (x_start, x_end) in suit_columns_ordered:
                if x_start <= x_orig < x_end:
                    detected_suit = suit_name
                    break
            
            if not detected_suit:
                # Default mapping based on equal-width columns
                col_width = w_img // 4
                if x_orig < col_width:
                    detected_suit = "spade"
                elif x_orig < col_width * 2:
                    detected_suit = "heart"
                elif x_orig < col_width * 3:
                    detected_suit = "diamond"
                else:
                    detected_suit = "club"
            
            # Crop card centered on peak (same as player cards)
            x_card = max(0, p["x_suit"] - 15)
            card_crop = dummy_area_norm[0:target_h, x_card:min(x_card + 40, w_norm)]
            
            # Extract rank using OCR
            rank_crop = card_crop[9:35, 6:24] if target_h == 60 else card_crop[2:int(target_h*0.43), 5:int(40*0.45)]
            
            rank_text = None
            for fx_val in [5.0, 4.0, 3.0]:
                gray_crop = cv2.cvtColor(rank_crop, cv2.COLOR_BGR2GRAY)
                scaled = cv2.resize(gray_crop, (0, 0), fx=fx_val, fy=fx_val, interpolation=cv2.INTER_CUBIC)
                thresh_crop = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                
                try:
                    txt = pytesseract.image_to_string(thresh_crop, config="--psm 10 -c tessedit_char_whitelist=AKQJT1098765432")
                    rank_text = clean_rank_candidate(txt)
                    if rank_text:
                        break
                except Exception:
                    pass
            
            contour_dummy_cards.append({
                "rank": rank_text,
                "suit": detected_suit,
                "cx": x_orig + 20,
                "cy": 125,
                "bbox": {"x": int(x_card / scale), "y": 0, "w": int(40 / scale), "h": int(target_h / scale)}
            })
    detected_cards.extend(contour_dummy_cards)
    
    # 3. Regional OCR candidate scanning for East/West dummy cards
    # We crop the left (West) and right (East) columns to avoid full-screen OCR
    raw_candidates = []
    
    # West area crop: x=0..110, y=180..600
    # East area crop: x=400..w_img, y=180..600
    crops = []
    if h_img >= 600:
        if w_img >= 110:
            crops.append(("West", img[180:600, 0:110], 0, 180))
        if w_img >= 400:
            crops.append(("East", img[180:600, 400:w_img], 400, 180))
        
    for side, crop, offset_x, offset_y in crops:
        if crop.size == 0:
            continue
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        for thresh_val, invert, psm in [(200, True, 11)]:
            _, thresh = cv2.threshold(gray_crop, thresh_val, 255, cv2.THRESH_BINARY)
            proc = cv2.bitwise_not(thresh) if invert else thresh
            scaled = cv2.resize(proc, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_NEAREST)
            
            try:
                data_str = pytesseract.image_to_data(scaled, config=f"--psm {psm}", output_type=pytesseract.Output.STRING)
                f = StringIO(data_str)
                reader = csv.reader(f, delimiter='\t')
                header = next(reader)
                
                left_idx = header.index('left')
                top_idx = header.index('top')
                width_idx = header.index('width')
                height_idx = header.index('height')
                text_idx = header.index('text')
                conf_idx = header.index('conf')
                
                for row in reader:
                    if len(row) <= text_idx:
                        continue
                    text = row[text_idx].strip()
                    if not text:
                        continue
                    conf = float(row[conf_idx])
                    if conf < 10:
                        continue
                    cleaned = clean_rank_candidate(text)
                    if cleaned:
                        cx = (int(row[left_idx]) + int(row[width_idx]) // 2) / 3.0 + offset_x
                        cy = (int(row[top_idx]) + int(row[height_idx]) // 2) / 3.0 + offset_y
                        raw_candidates.append((cleaned, cx, cy, conf))
            except Exception:
                pass
                
    unique_candidates = []
    for cand in raw_candidates:
        cleaned, cx, cy, conf = cand
        duplicate = False
        for idx, (uc_clean, uc_cx, uc_cy, uc_conf) in enumerate(unique_candidates):
            if abs(cx - uc_cx) < 15 and abs(cy - uc_cy) < 15:
                duplicate = True
                if conf > uc_conf:
                    unique_candidates[idx] = (cleaned, cx, cy, conf)
                break
        if not duplicate:
            unique_candidates.append((cleaned, cx, cy, conf))
            
    for rank_hint, cx, cy, conf in unique_candidates:
        card_w, card_h = 42, 66
        card_x1 = int(cx - 21)
        card_y1 = int(cy - 33)
        card_x2 = card_x1 + card_w
        card_y2 = card_y1 + card_h
        
        card_x1 = max(0, min(card_x1, w_img - 1))
        card_y1 = max(0, min(card_y1, h_img - 1))
        card_x2 = max(0, min(card_x2, w_img))
        card_y2 = max(0, min(card_y2, h_img))
        
        if (card_x2 - card_x1) < 20 or (card_y2 - card_y1) < 30:
            continue
            
        card_crop = img[card_y1:card_y2, card_x1:card_x2]
        rank, suit = analyzer.extract_card(card_crop, is_hand=False)
        
        if not rank and suit:
            rank = rank_hint
            
        if rank and suit:
            detected_cards.append({
                "rank": rank,
                "suit": suit,
                "cx": cx,
                "cy": cy,
                "bbox": {"x": card_x1, "y": card_y1, "w": card_x2 - card_x1, "h": card_y2 - card_y1}
            })
            
    west_dummy = []
    east_dummy = []
    north_dummy = []
    
    for card in detected_cards:
        cx, cy = card["cx"], card["cy"]
        if cy < 0.22 * h_img:
            north_dummy.append(card)
        elif cx < 0.22 * w_img and 0.22 * h_img <= cy < 0.75 * h_img:
            west_dummy.append(card)
        elif cx >= 0.78 * w_img and 0.22 * h_img <= cy < 0.75 * h_img:
            east_dummy.append(card)
            
    return {"West": west_dummy, "North": north_dummy, "East": east_dummy}

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
            flat = [b[1] for b in bids]
            print(" -> ".join(flat))
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

        # 4. Bidding Hint Text (above bidding headers)
        hint_img = cap.capture_bidding_hint()
        hint_text = analyzer.extract_bidding_hint(hint_img)
        print("\n--- BIDDING HINT ---")
        if hint_text:
            print(f"  {hint_text}")
        else:
            print("  (none)")
            
        # 5. Dummy Hands (West, North, East)
        ui_img = cap.capture_ui()
        print("\n--- DUMMY HANDS ---")
        dummy_hands = detect_dummy_hands(ui_img, analyzer)
        for side in ["West", "North", "East"]:
            hand = dummy_hands[side]
            print(f"  {side} Dummy Hand ({len(hand)} cards):")
            if hand:
                suits = {"spade": [], "heart": [], "diamond": [], "club": []}
                for c in hand:
                    if c["rank"] and c["suit"] in suits:
                        suits[c["suit"]].append(c["rank"])
                suit_symbols = {"spade": "♠", "heart": "♥", "diamond": "♦", "club": "♣"}
                parts = []
                for s in ["spade", "heart", "diamond", "club"]:
                    if suits[s]:
                        parts.append(f"{suit_symbols[s]}: {', '.join(suits[s])}")
                print("    " + " | ".join(parts))
            else:
                print("    None")
            
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run calibration first: python main.py --calibrate")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")

def run_click():
    """Executes the mouse automation sequence to click the contract button (level and suit) above the card area."""
    print("🖱️ Running click contract button automation...")
    try:
        ctrl = BridgeController()
        ctrl.click_build_info(return_to_start=True)
        print("✅ Click action performed successfully!")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run calibration first: python main.py --calibrate")
    except Exception as e:
        print(f"❌ Click action failed: {e}")

def run_bid_explanation(verbose=False):
    """
    Locates the last bid, clicks it, captures the explanation tooltip,
    runs OCR on it, cleans it up, and prints the explanation.
    """
    print("🔍 Attempting to click last bid and extract its explanation...")
    try:
        cap = ScreenCapture()
        analyzer = BridgeAnalyzer(verbose=verbose)
        ctrl = BridgeController()
        
        # 1. Capture Bidding ROI
        bidding_img = cap.capture_bidding()
        bids_with_bboxes = analyzer.extract_bids_with_bboxes(bidding_img)
        
        if not bids_with_bboxes:
            print("❌ No bids detected in bidding area.")
            return
            
        last_bid_obj = bids_with_bboxes[-1]
        print(f"👉 Last bid detected: {last_bid_obj['direction']}:{last_bid_obj['bid']}")
        
        # 2. Click the bid to trigger popup
        # Calculate global coordinates
        bidding_roi = cap.config["bidding_roi"]
        
        # Click the bid (returning mouse is True by default)
        target_x, target_y = ctrl.click_bid(last_bid_obj["bbox"], return_to_start=True)
        
        # 3. Capture a 350x200 crop centered on the click
        # Clamp coordinates within screen size
        try:
            import pyautogui
            screen_w, screen_h = pyautogui.size()
        except Exception:
            screen_w, screen_h = 1920, 1080
            
        crop_w, crop_h = 350, 200
        crop_x = max(0, min(screen_w - crop_w, target_x - crop_w // 2))
        crop_y = max(0, min(screen_h - crop_h, target_y - crop_h // 2))
        
        # Grab screen crop using mss
        monitor = {
            "top": int(crop_y),
            "left": int(crop_x),
            "width": int(crop_w),
            "height": int(crop_h)
        }
        
        # Capture crop
        screenshot = cap.sct.grab(monitor)
        img = np.array(screenshot)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # macOS Retina scale check
        h_img, w_img = img_bgr.shape[:2]
        if w_img != crop_w or h_img != crop_h:
            img_bgr = cv2.resize(img_bgr, (crop_w, crop_h), interpolation=cv2.INTER_AREA)
            
        # 4. Perform OCR on the crop
        processed = analyzer.preprocess_for_ocr(img_bgr, fx=4.0, thresh_val=None)
        
        # Run OCR with standard PSM 6
        text = pytesseract.image_to_string(processed, config="--psm 6")
        
        # 5. Clean up text
        # Filter out common headers, the bid itself, and empty lines
        lines = text.split("\n")
        cleaned_lines = []
        
        # Filter terms
        filter_terms = ["WEST", "NORTH", "EAST", "SOUTH", "BIDDING", "HISTORY", "YOUR", "HAND", "TRICK", "AREA", "BUILD", "INFO", "PASS", "DBL", "RDBL"]
        # Add directions N, E, S, W
        filter_terms.extend(["N", "E", "S", "W"])
        # Add the bid itself
        filter_terms.append(last_bid_obj["bid"].upper())
        
        for line in lines:
            line_cleaned = line.strip()
            if not line_cleaned:
                continue
                
            # If line is exactly a bidding table column header or just standard bids, skip
            words = line_cleaned.upper().split()
            if all(w in filter_terms or (len(w) >= 2 and w[0].isdigit() and w[1:] in ["C", "D", "H", "S", "NT"]) for w in words):
                continue
                
            cleaned_lines.append(line_cleaned)
            
        explanation = "\n".join(cleaned_lines).strip()
        
        print("\n====================================================")
        print("              BID EXPLANATION RESULTS               ")
        print("====================================================")
        if explanation:
            print(explanation)
        else:
            print("No new text explanation detected in tooltip popup.")
            print("Raw captured text in area:")
            print("----------------------------------------------------")
            print(text.strip())
        print("====================================================")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run calibration first: python main.py --calibrate")
    except Exception as e:
        print(f"❌ Bid explanation action failed: {e}")

def run_monitoring(interval=2.0, verbose=False, save_play=False, output_dir="captured_plays"):
    """Monitors the screen for changes in bids/cards and reports them."""
    print(f"👁️ Starting bridge play UI monitor (polling every {interval}s)...")
    print("Press Ctrl+C to stop.")
    
    tracker = GameTracker() if save_play else None
    
    try:
        cap = ScreenCapture()
        analyzer = BridgeAnalyzer(verbose=verbose)
        ctrl = BridgeController()
        
        last_bids = []
        prev_bidding_img = None
        prev_trick_img = None
        prev_hand_img = None
        bids = []
        detected_trick = []
        valid_hand = []
        
        while True:
            # Capture and extract bids
            bidding_img = cap.capture_bidding()
            if prev_bidding_img is not None and images_are_similar(prev_bidding_img, bidding_img, threshold=1.0):
                # Similar image, reuse bids
                pass
            else:
                bids = analyzer.extract_bids(bidding_img)
                prev_bidding_img = bidding_img.copy() if bidding_img is not None else None
            
            if save_play:
                tracker.update_bids(bids)
            
            # Check for changes in bids
            if bids != last_bids:
                print(f"\n📢 Bids changed! {time.strftime('%H:%M:%S')}", flush=True)
                prev_str = " -> ".join([f"{direction}:{bid}" for direction, bid in last_bids]) if last_bids else "None"
                curr_str = " -> ".join([f"{direction}:{bid}" for direction, bid in bids]) if bids else "None"
                print(f"Previous: {prev_str}", flush=True)
                print(f"Current : {curr_str}", flush=True)
                last_bids = bids
                
            # Perform trick check
            trick_img = cap.capture_trick()
            if prev_trick_img is not None and images_are_similar(prev_trick_img, trick_img, threshold=1.0):
                # Similar image, reuse detected_trick
                pass
            else:
                detected_trick = analyzer.extract_multiple_cards(trick_img)
                prev_trick_img = trick_img.copy() if trick_img is not None else None
                
            if save_play and trick_img is not None:
                tracker.register_trick_state(detected_trick, trick_img.shape[1], trick_img.shape[0])
                
            trick_str = ", ".join([f"{c['rank'] or '?'}{c['suit'] or '?'}" for c in detected_trick]) if detected_trick else "None"
            
            # Determine if we are in play stage (has trick cards or bidding has ended)
            has_trick = len(detected_trick) > 0
            flat_bids = [b[1] for b in bids]
            bidding_ended = False
            if flat_bids:
                consecutive_passes = 0
                for b in reversed(flat_bids):
                    if b.upper() == "PASS":
                        consecutive_passes += 1
                    else:
                        break
                bidding_ended = (consecutive_passes >= 3) or len(flat_bids) >= 40
            
            is_play_stage = has_trick or bidding_ended
            
            if is_play_stage:
                ui_img = cap.capture_ui()
                dummy_hands = detect_dummy_hands(ui_img, analyzer)
            else:
                dummy_hands = {"West": [], "North": [], "East": []}
                
            dummy_parts = []
            for side in ["West", "North", "East"]:
                d_hand_str = format_compact_hand(dummy_hands[side])
                if d_hand_str:
                    dummy_parts.append(f"{side}={d_hand_str}")
            dummy_str = " | ".join(dummy_parts) if dummy_parts else "None"
            
            sys.stdout.write(f"\rTrick: [{trick_str}] | Dummies: [{dummy_str}]   ")
            sys.stdout.flush()
                
            # Track player hand only if saving play
            if save_play:
                hand_img = cap.capture_player_hand()
                if prev_hand_img is not None and images_are_similar(prev_hand_img, hand_img, threshold=1.0):
                    pass
                else:
                    hand = analyzer.extract_hand_cards(hand_img)
                    valid_hand = [c for c in hand if c.get("rank") is not None and c.get("suit") is not None]
                    prev_hand_img = hand_img.copy() if hand_img is not None else None
                    
                tracker.set_initial_hand(valid_hand)
                
                # Check if board has finished (initial hand was set, but now hand is empty)
                if tracker.initial_hand and not valid_hand:
                    pbn_path, json_path = tracker.save_to_files(output_dir)
                    if pbn_path:
                        print(f"\n💾 Play saved successfully:\n   - {pbn_path}\n   - {json_path}", flush=True)
                    # Reset for next game
                    tracker = GameTracker()
                    prev_hand_img = None
                    prev_trick_img = None
                    prev_bidding_img = None
                
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
        if save_play and tracker:
            pbn_path, json_path = tracker.save_to_files(output_dir)
            if pbn_path:
                print(f"💾 Partially captured play saved to:\n   - {pbn_path}\n   - {json_path}")
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
            suits[suit].append(rank)
    
    rank_order = {r: idx for idx, r in enumerate(["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"])}
    
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

NEXT_DIR = {"N": "E", "E": "S", "S": "W", "W": "N"}

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
        
        b_clean = bid.upper()
        if b_clean == "PASS":
            b_compact = "P"
        elif b_clean == "DBL":
            b_compact = "X"
        elif b_clean == "RDBL":
            b_compact = "XX"
        else:
            b_compact = bid
            
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
        
    waiting = None
    if any(val != "-" for val in current_row):
        rows.append(current_row)
        if sum(1 for v in current_row if v == "-") > 0:
            last_dir = direction_bids[-1][0]
            waiting = NEXT_DIR.get(last_dir)
            
    lines = []
    lines.append("   N      E      S      W")
    lines.append("=========================")
    for r in rows:
        line_parts = []
        for i, d in enumerate(cols):
            val = r[i]
            if waiting and r is rows[-1] and val == "-":
                line_parts.append("?   " if i == cols.index(waiting) else "-   ")
            else:
                line_parts.append(f"{val:<4}")
        lines.append("  " + "   ".join(line_parts))
    if waiting:
        lines.append(f"⏳ Waiting for {waiting} to bid...")
    return "\n".join(lines)

def run_decision_loop(interval=2.0, dry_run=False, verbose=False, once=False, save_play=False, output_dir="captured_plays"):
    """Runs the capture -> analyze -> decide -> execute loop continuously or once."""
    mode = "single pass" if once else "continuous"
    print(f"🤖 Starting Bridge Play Decision Engine ({mode}, polling every {interval}s, dry_run={dry_run})...")
    if not once:
        print("Press Ctrl+C to stop.")
    
    tracker = GameTracker() if save_play else None
    
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
        last_hinted_state = None
        
        prev_bidding_img = None
        prev_hand_img = None
        prev_trick_img = None
        
        bids = []
        hand = []
        valid_hand = []
        trick_cards = []
        current_stage = None
        
        while True:
            current_time = time.time()
            
            # 1. Capture Bids
            bidding_img = cap.capture_bidding()
            if prev_bidding_img is not None and images_are_similar(prev_bidding_img, bidding_img, threshold=1.0):
                pass
            else:
                bids = analyzer.extract_bids(bidding_img)
                prev_bidding_img = bidding_img.copy() if bidding_img is not None else None
                
            if save_play:
                tracker.update_bids(bids)
            
            # 2. Capture Hand
            hand_img = cap.capture_player_hand()
            if prev_hand_img is not None and images_are_similar(prev_hand_img, hand_img, threshold=1.0):
                pass
            else:
                hand = analyzer.extract_hand_cards(hand_img)
                valid_hand = [c for c in hand if c.get("rank") is not None and c.get("suit") is not None]
                prev_hand_img = hand_img.copy() if hand_img is not None else None
                
            if save_play:
                tracker.set_initial_hand(valid_hand)
            
            # 3. Capture Trick Cards
            trick_img = cap.capture_trick()
            if prev_trick_img is not None and images_are_similar(prev_trick_img, trick_img, threshold=1.0):
                pass
            else:
                trick_cards = analyzer.extract_multiple_cards(trick_img)
                prev_trick_img = trick_img.copy() if trick_img is not None else None
                
            if save_play and trick_img is not None:
                tracker.register_trick_state(trick_cards, trick_img.shape[1], trick_img.shape[0])
            
            # 4. Determine Game Stage
            # Bidding is active if bidding hasn't ended (less than 3 passes) AND no cards are played in the trick area.
            has_trick_cards = len(trick_cards) > 0
            flat_bids = [b[1] for b in bids]
            bidding_ended = False
            if flat_bids:
                consecutive_passes = 0
                for b in reversed(flat_bids):
                    if b == "PASS":
                        consecutive_passes += 1
                    else:
                        break
                bidding_ended = (consecutive_passes >= 3) or len(flat_bids) >= 40
            
            is_bidding_active = not bidding_ended and not has_trick_cards and (len(hand) >= 13)
            
            # Log stage transitions
            detected_stage = "Bidding" if is_bidding_active else "Play"
            if detected_stage != current_stage:
                print(f"\n🔄 [Stage transition: {detected_stage}]", flush=True)
                current_stage = detected_stage
                
            # Output detection summary for this loop iteration
            flat_bids = [b[1] for b in bids]
            bids_str = format_compact_bids(flat_bids)
            hand_str = format_compact_hand(valid_hand)
            suit_symbols = {"spade": "♠", "heart": "♥", "diamond": "♦", "club": "♣"}
            trick_str = " ".join([f"{c['rank']}{suit_symbols.get(c['suit'], '')}" for c in trick_cards])
            
            # Detect dummy hands
            if detected_stage == "Play":
                ui_img = cap.capture_ui()
                dummy_hands = detect_dummy_hands(ui_img, analyzer)
            else:
                dummy_hands = {"West": [], "North": [], "East": []}
                
            dummy_parts = []
            for side in ["West", "North", "East"]:
                d_hand_str = format_compact_hand(dummy_hands[side])
                if d_hand_str:
                    dummy_parts.append(f"{side}={d_hand_str}")
            dummy_str = " | ".join(dummy_parts) if dummy_parts else "None"
            
            print(f"📸 Scan: Stage={detected_stage} | Bids=[{bids_str}] | Hand=[{hand_str}] | Trick=[{trick_str}] | Dummies=[{dummy_str}]", flush=True)

            # Capture and display hint text (runs every scan)
            if is_bidding_active:
                hint_img = cap.capture_bidding_hint()
                hint_text = analyzer.extract_bidding_hint(hint_img)
                if hint_text:
                    print(f"💡 Hint: {hint_text}", flush=True)

            # If we have no cards left, the board is finished
            if not valid_hand:
                if current_stage != "Waiting":
                    print("⏳ Board inactive or hand empty. Waiting...", flush=True)
                    current_stage = "Waiting"
                # Save captured play if tracker has data
                if save_play and tracker.initial_hand:
                    pbn_path, json_path = tracker.save_to_files(output_dir)
                    if pbn_path:
                        print(f"\n💾 Play saved successfully:\n   - {pbn_path}\n   - {json_path}", flush=True)
                    # Reset tracker for the next game
                    tracker = GameTracker()
                # Reset prev images to force re-evaluation when cards reappear
                prev_hand_img = None
                prev_trick_img = None
                prev_bidding_img = None
                last_hinted_state = None
                if once:
                    print("\nSingle-pass mode complete (no active hand detected).", flush=True)
                    break
                time.sleep(interval)
                continue
                
            if is_bidding_active:
                # --- BIDDING STATE ---
                
                if bids != last_bids:
                    last_bids = bids
                    table_str = format_bidding_table(bids)
                    print(f"\n📢 Bids updated:\n{table_str}")
                    
                suggested_bid = decide_bid(valid_hand, flat_bids)
                
                # Hint click automation
                if suggested_bid:
                    bids_count = len(flat_bids)
                    current_state = (bids_count, suggested_bid)
                    
                    if current_state != last_hinted_state:
                        if suggested_bid in ["PASS", "DBL", "RDBL"]:
                            print(f"🎯 Automating hint click for special bid: {suggested_bid}")
                            ui_roi = cap.config.get("ui_roi")
                            ui_img = cap.capture_ui()
                            hand_roi = cap.config.get("player_hand_roi")
                            max_y = hand_roi["y"] if hand_roi else None
                            
                            coords = analyzer.locate_ui_text_button(ui_img, suggested_bid, ui_roi, max_y=max_y)
                            if coords:
                                cx, cy = coords
                                if dry_run:
                                    print(f"🤖 [DRY RUN] Would move to special bid '{suggested_bid}' button at ({cx}, {cy}) and click.")
                                else:
                                    import pyautogui
                                    start_x, start_y = pyautogui.position()
                                    pyautogui.moveTo(cx, cy, duration=0.3, tween=pyautogui.easeInOutQuad)
                                    time.sleep(0.1)
                                    pyautogui.mouseDown()
                                    time.sleep(0.15)
                                    pyautogui.mouseUp()
                                    print(f"🖱️ Clicked special bid '{suggested_bid}' button.")
                                    time.sleep(3.0)
                                    pyautogui.moveTo(start_x, start_y, duration=0.3, tween=pyautogui.easeInOutQuad)
                                last_hinted_state = current_state
                        else:
                            # Standard suit bid, e.g., '1S', '2H', '1NT', '3NT'
                            level = suggested_bid[0]
                            suit = suggested_bid[1:]
                            
                            print(f"🎯 Automating hint click for bid: {suggested_bid}")
                            ui_roi = cap.config.get("ui_roi")
                            ui_img = cap.capture_ui()
                            hand_roi = cap.config.get("player_hand_roi")
                            max_y = hand_roi["y"] if hand_roi else None
                            
                            level_coords = analyzer.locate_ui_text_button(ui_img, level, ui_roi, max_y=max_y)
                            if level_coords:
                                lx, ly = level_coords
                                if dry_run:
                                    print(f"🤖 [DRY RUN] Would move to level '{level}' button at ({lx}, {ly}) and click.")
                                else:
                                    import pyautogui
                                    start_x, start_y = pyautogui.position()
                                    pyautogui.moveTo(lx, ly, duration=0.3, tween=pyautogui.easeInOutQuad)
                                    time.sleep(0.1)
                                    pyautogui.mouseDown()
                                    time.sleep(0.15)
                                    pyautogui.mouseUp()
                                    time.sleep(0.4) # Wait for animation / suits to display
                                    
                                # Re-capture UI to locate "NT" button
                                ui_img = cap.capture_ui()
                                nt_coords = analyzer.locate_ui_text_button(ui_img, "NT", ui_roi, max_y=max_y)
                                if nt_coords:
                                    nx, ny = nt_coords
                                    
                                    # Calculate suit offset
                                    # NT: 0, S: -50, H: -100, D: -150, C: -200
                                    suit_offsets = {
                                        "NT": 0,
                                        "S": -50,
                                        "H": -100,
                                        "D": -150,
                                        "C": -200
                                    }
                                    offset = suit_offsets.get(suit, 0)
                                    tx, ty = nx + offset, ny
                                    
                                    if dry_run:
                                        print(f"🤖 [DRY RUN] Would move to suit '{suit}' button at ({tx}, {ty}) and click.")
                                    else:
                                        pyautogui.moveTo(tx, ty, duration=0.3, tween=pyautogui.easeInOutQuad)
                                        time.sleep(0.1)
                                        pyautogui.mouseDown()
                                        time.sleep(0.15)
                                        pyautogui.mouseUp()
                                        print(f"🖱️ Clicked suit '{suit}' to show hint.")
                                        time.sleep(3.0) # Wait 3s to let the user see the hint
                                        pyautogui.moveTo(start_x, start_y, duration=0.3, tween=pyautogui.easeInOutQuad)
                                last_hinted_state = current_state
            else:
                # --- PLAY STATE ---
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
                            
                # If we played a card but it's been more than 5.0 seconds and hand size hasn't changed,
                # the click was likely ignored or missed, so reset played flag to retry.
                if played_in_current_trick and current_time - last_action_time > 5.0:
                    if len(valid_hand) == last_hand_len:
                        played_in_current_trick = False
                        print("⚠️ Play timeout: Hand size did not decrease after 5.0s. Retrying...", flush=True)
                
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
        if save_play and tracker:
            pbn_path, json_path = tracker.save_to_files(output_dir)
            if pbn_path:
                print(f"💾 Partially captured play saved to:\n   - {pbn_path}\n   - {json_path}")
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
    group.add_argument("--click", action="store_true", help="Click the contract button (number and suit) using mouse automation")
    group.add_argument("--monitor", action="store_true", help="Start continuous bridge UI state monitor")
    group.add_argument("--run", action="store_true", help="Start continuous bridge play & decision runner")
    group.add_argument("--explain-last-bid", action="store_true", help="Click the last bid in history to extract its tooltip explanation")
    
    parser.add_argument("--interval", type=float, default=2.0, help="Polling interval for monitoring in seconds (default: 2.0)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (logs decisions and actions without moving mouse)")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode (logs detailed computer vision and template matching details)")
    parser.add_argument("--once", action="store_true", help="Run one iteration and exit (useful with --run)")
    parser.add_argument("--save-play", action="store_true", help="Save the captured bids, hand, and play sequence into PBN and JSON files")
    parser.add_argument("--output-dir", type=str, default="captured_plays", help="Directory where captured plays are stored (default: 'captured_plays')")

    args = parser.parse_args()

    if args.calibrate:
        run_calibration()
    elif args.capture_debug:
        run_capture_debug()
    elif args.analyze:
        run_analysis(verbose=args.verbose)
    elif args.click:
        run_click()
    elif args.explain_last_bid:
        run_bid_explanation(verbose=args.verbose)
    elif args.monitor:
        run_monitoring(args.interval, verbose=args.verbose, save_play=args.save_play, output_dir=args.output_dir)
    elif args.run:
        run_decision_loop(args.interval, args.dry_run, verbose=args.verbose, once=args.once, save_play=args.save_play, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
