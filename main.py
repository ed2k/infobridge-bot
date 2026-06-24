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


class RegionDeltaDetector:
    """
    Tracks pixel-level changes in sub-regions of captured images.
    Only triggers re-detection when a region has changed significantly.
    """

    def __init__(self, change_threshold=2.0, min_change_pct=0.5):
        """
        Args:
            change_threshold: Per-pixel MAE threshold to consider a pixel "changed".
            min_change_pct: Minimum percentage of changed pixels to trigger re-detection.
        """
        self.change_threshold = change_threshold
        self.min_change_pct = min_change_pct
        self._prev_gray = {}

    def _to_gray(self, img):
        if img is None:
            return None
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def region_changed(self, key, img, roi=None):
        """
        Check if a region has changed since the last call.

        Args:
            key: Unique identifier for this region (e.g. "bidding", "trick", "hand").
            img: Current BGR or grayscale image.
            roi: Optional dict {x, y, width, height} to extract sub-region from img.

        Returns:
            True if the region has changed enough to warrant re-detection.
        """
        if img is None:
            return True

        if roi is not None:
            x, y = int(roi["x"]), int(roi["y"])
            w, h = int(roi["width"]), int(roi["height"])
            img = img[y:y+h, x:x+w]

        gray = self._to_gray(img)
        if gray is None:
            return True

        prev = self._prev_gray.get(key)
        self._prev_gray[key] = gray.copy()

        if prev is None:
            return True

        if prev.shape != gray.shape:
            return True

        diff = cv2.absdiff(prev, gray)
        changed_pixels = np.sum(diff > self.change_threshold)
        total_pixels = diff.size
        change_pct = (changed_pixels / total_pixels) * 100

        return change_pct >= self.min_change_pct

    def reset(self, key=None):
        """Clear cached images. If key is None, clear all."""
        if key is None:
            self._prev_gray.clear()
        else:
            self._prev_gray.pop(key, None)

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
        print("✅ Captured successfully. Look in the 'debug' folder to check boundaries.")
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

def _nms_boxes(boxes, threshold=0.3):
    """Non-Maximum Suppression to filter overlapping detections."""
    if len(boxes) == 0:
        return []
    
    # boxes format: (x, y, w, h, score, label)
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    pick = []
    
    while len(boxes) > 0:
        b = boxes[0]
        pick.append(b)
        boxes = boxes[1:]
        
        remaining = []
        for box in boxes:
            x1 = max(b[0], box[0])
            y1 = max(b[1], box[1])
            x2 = min(b[0] + b[2], box[0] + box[2])
            y2 = min(b[1] + b[3], box[1] + box[3])
            
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            intersection = w * h
            
            area_b = b[2] * b[3]
            area_box = box[2] * box[3]
            union = area_b + area_box - intersection
            
            iou = intersection / union if union > 0 else 0
            if iou < threshold:
                remaining.append(box)
        boxes = remaining
        
    return pick

def detect_dummy_hands(img, analyzer):
    """
    Detects dummy hands on all sides (West, East, North) in the full UI capture image.
    North dummy uses the same card detection as player hand (same card layout).
    East/West use 2D template matching for robust rank and suit pairing.
    """
    if img is None:
        return {"West": [], "North": [], "East": []}
        
    h_img, w_img = img.shape[:2]
    detected_cards = []
    
    # 1. North dummy: Same layout as player hand (one row of cards)
    # The dummy card strip is at y=240..340 in the UI
    if h_img >= 340:
        # Crop the center section where North dummy cards reside to avoid sidebar noise
        if w_img > 600:
            left_x = w_img // 2 - 250
            right_x = w_img // 2 + 250
            dummy_strip = img[240:340, left_x:right_x]
            offset_x = left_x
        else:
            dummy_strip = img[240:340, 0:w_img]
            offset_x = 0
            
        try:
            import os
            os.makedirs("debug", exist_ok=True)
            cv2.imwrite("debug/dummy_strip_north.png", dummy_strip)
            
            # Strict white pixel ratio check to filter out card backs or empty background
            hsv_dummy = cv2.cvtColor(dummy_strip, cv2.COLOR_BGR2HSV)
            white_pixels = np.sum((hsv_dummy[:,:,1] < 30) & (hsv_dummy[:,:,2] > 200))
            white_ratio = white_pixels / (dummy_strip.shape[0] * dummy_strip.shape[1])
            
            if white_ratio >= 0.15:
                north_cards = analyzer.extract_hand_cards(dummy_strip)
                # Adjust coordinates to full image
                for card in north_cards:
                    if card.get("rank") and card.get("suit"):
                        bbox = card.get("bbox", {})
                        detected_cards.append({
                            "rank": card["rank"],
                            "suit": card["suit"],
                            "cx": offset_x + bbox.get("x", 0) + bbox.get("w", 0) // 2,
                            "cy": 240 + bbox.get("y", 0) + bbox.get("h", 0) // 2,
                            "side": "North",
                            "bbox": {
                                "x": bbox.get("x", 0),
                                "y": bbox.get("y", 0) + 240,
                                "w": bbox.get("w", 0),
                                "h": bbox.get("h", 0)
                            }
                        })
        except Exception:
            pass
            
    # 2. East/West dummy: Template Matching
    if not hasattr(analyzer, 'rank_templates'):
        analyzer.rank_templates = {}
        ranks = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
        for r in ranks:
            p = os.path.join(analyzer.templates_dir, f"rank_{r}.png")
            if os.path.exists(p):
                analyzer.rank_templates[r] = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            else:
                p2 = os.path.join(analyzer.templates_dir, f"{r}.png")
                if os.path.exists(p2):
                    analyzer.rank_templates[r] = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)

    crops = []
    if h_img >= 600:
        if w_img >= 110:
            west_crop = img[320:min(620, h_img), 0:110]
            crops.append(("West", west_crop, 0, 320))
            cv2.imwrite("debug/dummy_strip_west.png", west_crop)
        if w_img >= 400:
            east_crop = img[320:min(620, h_img), 400:w_img]
            crops.append(("East", east_crop, 400, 320))
            cv2.imwrite("debug/dummy_strip_east.png", east_crop)

    for side, crop, offset_x, offset_y in crops:
        if crop.size == 0:
            continue
            
        # Strict white pixel ratio check to filter out card backs or empty background
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        white_pixels = np.sum((hsv_crop[:,:,1] < 30) & (hsv_crop[:,:,2] > 200))
        white_ratio = white_pixels / (crop.shape[0] * crop.shape[1])
        
        if white_ratio < 0.15:
            continue
            
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Get PaddleOCR text recognition on the side crop
        paddle_results = []
        try:
            from paddle_ocr import ocr_image
            paddle_results = ocr_image(crop)
        except Exception:
            pass
        
        # Match ranks
        rank_detections = []
        for r, tpl in analyzer.rank_templates.items():
            if tpl is None:
                continue
            res = cv2.matchTemplate(gray_crop, tpl, cv2.TM_CCOEFF_NORMED)
            h_tpl, w_tpl = tpl.shape
            # Local maxima check
            dilated = cv2.dilate(res, np.ones((3, 3)))
            loc = np.where((res == dilated) & (res >= 0.72))
            for pt in zip(*loc[::-1]):
                patch = gray_crop[pt[1]:pt[1]+h_tpl, pt[0]:pt[0]+w_tpl]
                # Ensure the matched area is on a white card (contains white pixels)
                if np.max(patch) > 200:
                    # Quick check: does the matched patch contain dark symbol pixels?
                    if np.min(patch) < 135:
                        score = res[pt[1], pt[0]]
                        rank_detections.append((pt[0], pt[1], w_tpl, h_tpl, score, r))
                
        filtered_ranks = _nms_boxes(rank_detections, threshold=0.3)
        
        # Match suits (NMS per suit to prevent cross-suit suppression)
        filtered_suits = []
        for s, tpl in analyzer.suit_templates.items():
            if tpl is None:
                continue
            res = cv2.matchTemplate(gray_crop, tpl, cv2.TM_CCOEFF_NORMED)
            h_tpl, w_tpl = tpl.shape
            # Local maxima check
            dilated = cv2.dilate(res, np.ones((3, 3)))
            loc = np.where((res == dilated) & (res >= 0.75))
            s_detections = []
            for pt in zip(*loc[::-1]):
                patch = gray_crop[pt[1]:pt[1]+h_tpl, pt[0]:pt[0]+w_tpl]
                # Ensure the matched area is on a white card (contains white pixels)
                if np.max(patch) > 200:
                    # Quick check: does the matched patch contain dark symbol pixels?
                    if np.min(patch) < 135:
                        score = res[pt[1], pt[0]]
                        s_detections.append((pt[0], pt[1], w_tpl, h_tpl, score, s))
            filtered_suits.extend(_nms_boxes(s_detections, threshold=0.3))
        
        # Pair ranks and suits using spatial relationship (suit offset: dx [3, 13], dy [22, 38])
        paired_candidates = []
        used_suits = set()
        for rx, ry, rw, rh, r_score, r_label in filtered_ranks:
            best_suit_info = None
            best_suit_idx = -1
            min_dist = float('inf')
            
            for s_idx, (sx, sy, sw, sh, s_score, s_label) in enumerate(filtered_suits):
                if s_idx in used_suits:
                    continue
                dx = sx - rx
                dy = sy - ry
                if 3 <= dx <= 13 and 22 <= dy <= 38:
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_suit_info = (sx, sy, sw, sh, s_label)
                        best_suit_idx = s_idx
                        
            if best_suit_info:
                sx, sy, sw, sh, s_label = best_suit_info
                # Get the 13x13 suit crop
                sy_start = max(0, sy)
                sy_end = min(crop.shape[0], sy + 13)
                sx_start = max(0, sx)
                sx_end = min(crop.shape[1], sx + 13)
                suit_patch = crop[sy_start:sy_end, sx_start:sx_end]
                
                # Check color (is_red)
                is_red = False
                if suit_patch.size > 0:
                    hsv_patch = cv2.cvtColor(suit_patch, cv2.COLOR_BGR2HSV)
                    lower_red1 = np.array([0, 50, 50])
                    upper_red1 = np.array([25, 255, 255])
                    lower_red2 = np.array([170, 50, 50])
                    upper_red2 = np.array([180, 255, 255])
                    mask = cv2.inRange(hsv_patch, lower_red1, upper_red1) + cv2.inRange(hsv_patch, lower_red2, upper_red2)
                    is_red = (np.sum(mask > 0) / suit_patch.size) > 0.015
                
                # Calculate template matching scores for all 4 suits on the gray patch
                scores = {}
                gray_patch = gray_crop[sy_start:sy_end, sx_start:sx_end]
                if gray_patch.size > 0:
                    if gray_patch.shape[0] < 13 or gray_patch.shape[1] < 13:
                        gray_patch = cv2.resize(gray_patch, (13, 13))
                    for suit_name, tpl in analyzer.suit_templates.items():
                        if tpl is not None:
                            res_patch = cv2.matchTemplate(gray_patch, tpl, cv2.TM_CCOEFF_NORMED)
                            _, max_val, _, _ = cv2.minMaxLoc(res_patch)
                            scores[suit_name] = max_val
                
                paired_candidates.append({
                    "rx": rx, "ry": ry, "rw": rw, "rh": rh,
                    "sx": sx, "sy": sy,
                    "r_label": r_label,
                    "is_red": is_red,
                    "scores": scores,
                    "best_suit_idx": best_suit_idx
                })
                used_suits.add(best_suit_idx)

        # Group paired cards into rows based on ry (clustering within 25px)
        paired_candidates.sort(key=lambda x: x["ry"])
        rows = []
        for cand in paired_candidates:
            if not rows or cand["ry"] - rows[-1][0]["ry"] > 25:
                rows.append([cand])
            else:
                rows[-1].append(cand)
                
        # Classify the suit of each row using strict Bridge layout rules: Spade -> Heart -> Diamond -> Club
        n_rows = len(rows)
        row_colors = []
        for row in rows:
            is_red_votes = [c["is_red"] for c in row]
            row_colors.append(sum(is_red_votes) > len(is_red_votes) / 2)
            
        all_valid_sequences = [
            # Subsets of size 1
            ["spade"], ["heart"], ["diamond"], ["club"],
            # Subsets of size 2
            ["spade", "heart"], ["spade", "diamond"], ["spade", "club"],
            ["heart", "diamond"], ["heart", "club"], ["diamond", "club"],
            # Subsets of size 3
            ["spade", "heart", "diamond"], ["spade", "heart", "club"],
            ["spade", "diamond", "club"], ["heart", "diamond", "club"],
            # Subsets of size 4
            ["spade", "heart", "diamond", "club"]
        ]
        suit_colors = {"spade": False, "heart": True, "diamond": True, "club": False}
        
        candidates = []
        for seq in all_valid_sequences:
            if len(seq) == n_rows:
                seq_colors = [suit_colors[s] for s in seq]
                if seq_colors == row_colors:
                    candidates.append(seq)
                    
        if candidates:
            # Choose candidate sequence with the highest total template score
            best_seq = None
            max_total_score = -float('inf')
            for seq in candidates:
                total_score = 0
                for idx, suit_name in enumerate(seq):
                    total_score += sum(c["scores"].get(suit_name, 0) for c in rows[idx])
                if total_score > max_total_score:
                    max_total_score = total_score
                    best_seq = seq
            row_suits = best_seq
        else:
            # Fallback: independent classification
            row_suits = []
            for idx, row in enumerate(rows):
                is_red = row_colors[idx]
                if is_red:
                    heart_sum = sum(c["scores"].get("heart", 0) for c in row)
                    diamond_sum = sum(c["scores"].get("diamond", 0) for c in row)
                    row_suits.append("heart" if heart_sum > diamond_sum else "diamond")
                else:
                    spade_sum = sum(c["scores"].get("spade", 0) for c in row)
                    club_sum = sum(c["scores"].get("club", 0) for c in row)
                    row_suits.append("spade" if spade_sum > club_sum else "club")
                    
        # Process card assignments row by row
        for r_idx, row in enumerate(rows):
            row_suit = row_suits[r_idx]
            
            # Align row with PaddleOCR text boxes to override template matched ranks
            if row and paddle_results:
                row_cy = sum(c["ry"] for c in row) / len(row)
                matching_res = None
                for res in paddle_results:
                    bbox = res["bbox"]
                    cy = (bbox[0][1] + bbox[2][1]) / 2.0
                    if abs(cy - row_cy) < 20:
                        matching_res = res
                        break
                if matching_res:
                    def parse_ranks(text):
                        text = text.upper().replace(" ", "")
                        ranks = []
                        i = 0
                        while i < len(text):
                            if i + 1 < len(text) and text[i:i+2] == "10":
                                ranks.append("T")
                                i += 2
                            else:
                                char = text[i]
                                if char in "AKQJT98765432":
                                    ranks.append(char)
                                    i += 1
                                elif char == "1" and i + 1 < len(text) and text[i+1] == "0":
                                    ranks.append("T")
                                    i += 2
                                elif char == "1":
                                    ranks.append("T")
                                    i += 1
                                else:
                                    i += 1
                        return ranks
                        
                    ocr_ranks = parse_ranks(matching_res["text"])
                    if ocr_ranks:
                        # Sort row candidates horizontally from left to right
                        row.sort(key=lambda x: x["rx"])
                        x1 = min(pt[0] for pt in matching_res["bbox"])
                        x2 = max(pt[0] for pt in matching_res["bbox"])
                        w = x2 - x1
                        for cand in row:
                            best_ocr_rank = None
                            min_dx = float('inf')
                            for idx, r in enumerate(ocr_ranks):
                                est_x = x1 + (idx + 0.5) * (w / len(ocr_ranks))
                                dx = abs(cand["rx"] - est_x)
                                if dx < min_dx:
                                    min_dx = dx
                                    best_ocr_rank = r
                            # Override template rank if it matches horizontal position of OCR rank within 25px
                            if best_ocr_rank and min_dx < 25:
                                cand["r_label"] = best_ocr_rank
                                
            # Assign suit to each card in the row and add to detected_cards
            for cand in row:
                cx = cand["rx"] + cand["rw"] // 2 + offset_x
                cy = cand["ry"] + cand["rh"] // 2 + offset_y
                
                card_x1 = max(0, int(cx - 21))
                card_y1 = max(0, int(cy - 33))
                card_x2 = min(w_img, card_x1 + 42)
                card_y2 = min(h_img, card_y1 + 66)
                
                detected_cards.append({
                    "rank": cand["r_label"],
                    "suit": row_suit,
                    "cx": cx,
                    "cy": cy,
                    "side": side,
                    "bbox": {"x": card_x1, "y": card_y1, "w": card_x2 - card_x1, "h": card_y2 - card_y1}
                })
                
                # Save crop to debug folder
                rel_x1 = max(0, cand["rx"] + cand["rw"] // 2 - 21)
                rel_y1 = max(0, cand["ry"] + cand["rh"] // 2 - 33)
                rel_x2 = min(crop.shape[1], rel_x1 + 42)
                rel_y2 = min(crop.shape[0], rel_y1 + 66)
                card_crop = crop[rel_y1:rel_y2, rel_x1:rel_x2]
                if card_crop.size > 0:
                    cv2.imwrite(f"debug/dummy_card_{side}_{cand['r_label']}_{row_suit}.png", card_crop)

    west_dummy = []
    east_dummy = []
    north_dummy = []
    
    for card in detected_cards:
        side = card.get("side")
        if side == "West":
            west_dummy.append(card)
        elif side == "East":
            east_dummy.append(card)
        elif side == "North":
            north_dummy.append(card)
        else:
            # Fallback to coordinate check
            cx, cy = card["cx"], card["cy"]
            if 220 <= cy <= 360:
                north_dummy.append(card)
            elif cx < 0.22 * w_img and 0.22 * h_img <= cy < 0.75 * h_img:
                west_dummy.append(card)
            elif cx >= 0.78 * w_img and 0.22 * h_img <= cy < 0.75 * h_img:
                east_dummy.append(card)
                
    # Single-dummy-side heuristic: only one side can be the dummy hand at any given time.
    # Keep only the detections for the side with the maximum card count, clearing the others.
    counts = {
        "West": len(west_dummy),
        "North": len(north_dummy),
        "East": len(east_dummy)
    }
    best_side = max(counts, key=counts.get)
    if counts[best_side] > 0:
        if best_side != "West":
            west_dummy = []
        if best_side != "North":
            north_dummy = []
        if best_side != "East":
            east_dummy = []
            
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
        import os; os.makedirs("debug", exist_ok=True)
        cv2.imwrite("debug/player_hand_area.png", hand_img)
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
        delta = RegionDeltaDetector()
        
        last_bids = []
        bids = []
        detected_trick = []
        valid_hand = []
        
        while True:
            bidding_img = cap.capture_bidding()
            if delta.region_changed("bidding", bidding_img):
                bids = analyzer.extract_bids(bidding_img)
            
            if save_play:
                tracker.update_bids(bids)
            
            if bids != last_bids:
                print(f"\n📢 Bids changed! {time.strftime('%H:%M:%S')}", flush=True)
                prev_str = " -> ".join([f"{direction}:{bid}" for direction, bid in last_bids]) if last_bids else "None"
                curr_str = " -> ".join([f"{direction}:{bid}" for direction, bid in bids]) if bids else "None"
                print(f"Previous: {prev_str}", flush=True)
                print(f"Current : {curr_str}", flush=True)
                last_bids = bids
                
            trick_img = cap.capture_trick()
            if delta.region_changed("trick", trick_img):
                detected_trick = analyzer.extract_multiple_cards(trick_img)
                
            if save_play and trick_img is not None:
                tracker.register_trick_state(detected_trick, trick_img.shape[1], trick_img.shape[0])
                
            trick_str = ", ".join([f"{c['rank'] or '?'}{c['suit'] or '?'}" for c in detected_trick]) if detected_trick else "None"
            
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
                
            if save_play:
                hand_img = cap.capture_player_hand()
                if delta.region_changed("hand", hand_img):
                    hand = analyzer.extract_hand_cards(hand_img)
                    valid_hand = [c for c in hand if c.get("rank") is not None and c.get("suit") is not None]
                    
                tracker.set_initial_hand(valid_hand)
                
                if tracker.initial_hand and not valid_hand:
                    pbn_path, json_path = tracker.save_to_files(output_dir)
                    if pbn_path:
                        print(f"\n💾 Play saved successfully:\n   - {pbn_path}\n   - {json_path}", flush=True)
                    tracker = GameTracker()
                    delta.reset()
                
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

def run_decision_loop(interval=2.0, dry_run=False, verbose=False, once=False, save_play=False, output_dir="captured_plays", action=False):
    """Runs the capture -> analyze -> decide -> execute loop continuously or once."""
    mode = "single pass" if once else "continuous"
    print(f"🤖 Starting Bridge Play Decision Engine ({mode}, polling every {interval}s, dry_run={dry_run}, action={action})...")
    if not once:
        print("Press Ctrl+C to stop.")
    
    tracker = GameTracker() if save_play else None
    
    try:
        from strategy import decide_bid, decide_play_card
        cap = ScreenCapture()
        analyzer = BridgeAnalyzer(verbose=verbose)
        ctrl = BridgeController(dry_run=dry_run, action=action)
        delta = RegionDeltaDetector()
        
        last_bids = []
        last_trick_cards = []
        played_in_current_trick = False
        last_hand_len = 0
        last_action_time = 0
        COOLDOWN = 4.0
        last_hinted_state = None
        
        bids = []
        hand = []
        valid_hand = []
        trick_cards = []
        current_stage = None
        
        while True:
            current_time = time.time()
            
            bidding_img = cap.capture_bidding()
            if delta.region_changed("bidding", bidding_img):
                bids = analyzer.extract_bids(bidding_img)
                
            if save_play:
                tracker.update_bids(bids)
            
            hand_img = cap.capture_player_hand()
            import os; os.makedirs("debug", exist_ok=True)
            cv2.imwrite("debug/player_hand_area.png", hand_img)
            if delta.region_changed("hand", hand_img):
                hand = analyzer.extract_hand_cards(hand_img)
                valid_hand = [c for c in hand if c.get("rank") is not None and c.get("suit") is not None]
                
            if save_play:
                tracker.set_initial_hand(valid_hand)
            
            trick_img = cap.capture_trick()
            if delta.region_changed("trick", trick_img):
                trick_cards = analyzer.extract_multiple_cards(trick_img)
                
            if save_play and trick_img is not None:
                tracker.register_trick_state(trick_cards, trick_img.shape[1], trick_img.shape[0])
            
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
            
            if detected_stage == "Play":
                ui_img = cap.capture_ui()
                if delta.region_changed("ui", ui_img):
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

            if is_bidding_active:
                hint_img = cap.capture_bidding_hint()
                if delta.region_changed("hint", hint_img):
                    hint_text = analyzer.extract_bidding_hint(hint_img)
                    if hint_text:
                        print(f"💡 Hint: {hint_text}", flush=True)

            if not valid_hand:
                if current_stage != "Waiting":
                    print("⏳ Board inactive or hand empty. Waiting...", flush=True)
                    current_stage = "Waiting"
                if save_play and tracker.initial_hand:
                    pbn_path, json_path = tracker.save_to_files(output_dir)
                    if pbn_path:
                        print(f"\n💾 Play saved successfully:\n   - {pbn_path}\n   - {json_path}", flush=True)
                    tracker = GameTracker()
                delta.reset()
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
                                if not action:
                                    print(f"🤖 [NO ACTION] Would move to special bid '{suggested_bid}' button at ({cx}, {cy}) and click.")
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
                                if not action:
                                    print(f"🤖 [NO ACTION] Would move to level '{level}' button at ({lx}, {ly}) and click.")
                                else:
                                    import pyautogui
                                    start_x, start_y = pyautogui.position()
                                    pyautogui.moveTo(lx, ly, duration=0.3, tween=pyautogui.easeInOutQuad)
                                    time.sleep(0.1)
                                    pyautogui.mouseDown()
                                    time.sleep(0.15)
                                    pyautogui.mouseUp()
                                    time.sleep(0.4)
                                    
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
                                    
                                    if not action:
                                        print(f"🤖 [NO ACTION] Would move to suit '{suit}' button at ({tx}, {ty}) and click.")
                                    else:
                                        pyautogui.moveTo(tx, ty, duration=0.3, tween=pyautogui.easeInOutQuad)
                                        time.sleep(0.1)
                                        pyautogui.mouseDown()
                                        time.sleep(0.15)
                                        pyautogui.mouseUp()
                                        print(f"🖱️ Clicked suit '{suit}' to show hint.")
                                        time.sleep(3.0)
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
    group.add_argument("--capture-debug", action="store_true", help="Capture UI regions and save to debug/")
    group.add_argument("--analyze", action="store_true", help="Run OCR/CV analysis on the live screen once")
    group.add_argument("--click", action="store_true", help="Click the contract button (number and suit) using mouse automation")
    group.add_argument("--monitor", action="store_true", help="Start continuous bridge UI state monitor")
    group.add_argument("--run", action="store_true", help="Start continuous bridge play & decision runner")
    group.add_argument("--explain-last-bid", action="store_true", help="Click the last bid in history to extract its tooltip explanation")
    group.add_argument("--update-templates", action="store_true", help="Update rank templates from live screen capture")
    group.add_argument("--generate-mock", action="store_true", help="Generate sample_board.png from live capture or synthetic fallback")

    parser.add_argument("--interval", type=float, default=2.0, help="Polling interval for monitoring in seconds (default: 2.0)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (logs decisions and actions without moving mouse)")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode (logs detailed computer vision and template matching details)")
    parser.add_argument("--once", action="store_true", help="Run one iteration and exit (useful with --run)")
    parser.add_argument("--save-play", action="store_true", help="Save the captured bids, hand, and play sequence into PBN and JSON files")
    parser.add_argument("--output-dir", type=str, default="captured_plays", help="Directory where captured plays are stored (default: 'captured_plays')")
    parser.add_argument("--action", action="store_true", help="Enable actual mouse movement (default: observation only)")

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
        run_decision_loop(args.interval, args.dry_run, verbose=args.verbose, once=args.once, save_play=args.save_play, output_dir=args.output_dir, action=args.action)
    elif args.update_templates:
        import update_templates_live
        update_templates_live.main()
    elif args.generate_mock:
        import generate_mock
        generate_mock.main()

if __name__ == "__main__":
    main()
