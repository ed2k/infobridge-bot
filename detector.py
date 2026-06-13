#!/usr/bin/env python3
"""
Dynamic UI Detector for Bridge Bot.
Automatically detects game panel boundaries, bidding columns, hand card positions,
and suit icons from a live screen capture. Replaces hard-coded ROIs and pixel offsets.
"""

import cv2
import numpy as np
import re
import os
import csv
from io import StringIO
import pytesseract


class UIDetector:
    """Detects bridge UI elements dynamically from screen captures."""

    def __init__(self, verbose=False):
        self.verbose = verbose

    def detect_game_panel(self, screen_img, roi_hint=None):
        """
        Detect the main game panel in a full or partial screen capture.
        Uses the green felt table color and surrounding dark borders to find bounds.
        
        Args:
            screen_img: Full or partial BGR screen capture.
            roi_hint: Optional dict with x,y,width,height to narrow search area.
        
        Returns:
            dict with x, y, width, height of the detected game panel.
        """
        img = screen_img.copy()
        if roi_hint:
            hx, hy = roi_hint["x"], roi_hint["y"]
            img = img[hy:hy + roi_hint["height"], hx:hx + roi_hint["width"]]
            offset_x, offset_y = hx, hy
        else:
            offset_x, offset_y = 0, 0

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        lower_green = np.array([25, 30, 30])
        upper_green = np.array([95, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        kernel = np.ones((15, 15), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            if self.verbose:
                print("⚠️ UIDetector: No green felt detected, using full image bounds.")
            h, w = img.shape[:2]
            return {"x": offset_x, "y": offset_y, "width": w, "height": h}
        
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        
        pad = 10
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(img.shape[1] - x, w + 2 * pad)
        h = min(img.shape[0] - y, h + 2 * pad)
        
        if self.verbose:
            print(f"🔍 UIDetector: Game panel at ({x + offset_x}, {y + offset_y}, {w}x{h})")
        
        return {"x": x + offset_x, "y": y + offset_y, "width": w, "height": h}

    def detect_bidding_columns(self, bidding_img, fx=4.0):
        """
        Detect N/E/S/W column positions in the bidding table from OCR data.
        Returns ordered list of (direction, center_x) and the column width.
        """
        gray = cv2.cvtColor(bidding_img, cv2.COLOR_BGR2GRAY)
        scaled = cv2.resize(gray, (0, 0), fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
        
        data_str = pytesseract.image_to_data(
            scaled, config="--psm 6", output_type=pytesseract.Output.STRING
        )
        
        f = StringIO(data_str)
        reader = csv.reader(f, delimiter='\t')
        
        try:
            header = next(reader)
        except StopIteration:
            return [], 0
        
        try:
            left_idx = header.index('left')
            top_idx = header.index('top')
            width_idx = header.index('width')
            height_idx = header.index('height')
            text_idx = header.index('text')
        except ValueError:
            return [], 0
        
        direction_map = {"SOUTH": "S", "NORTH": "N", "EAST": "E", "WEST": "W"}
        header_words = []
        
        for row in reader:
            if len(row) <= text_idx:
                continue
            text = row[text_idx].strip()
            if not text:
                continue
            
            cleaned = re.sub(r'[^a-zA-Z]', '', text).upper()
            
            dir_key = None
            if cleaned in direction_map:
                dir_key = direction_map[cleaned]
            else:
                for full_word, direction in direction_map.items():
                    if full_word.startswith(cleaned):
                        dir_key = direction
                        break
            
            top_val = int(row[top_idx])
            img_h = scaled.shape[0]
            
            if dir_key and top_val < img_h * 0.35:
                cx = int(row[left_idx]) + int(row[width_idx]) // 2
                header_words.append((cx, dir_key, top_val))
        
        if len(header_words) < 2:
            return [], 0
        
        header_words.sort()
        
        min_top = min(hw[2] for hw in header_words)
        filtered = [hw for hw in header_words if hw[2] - min_top < 30]
        
        if len(filtered) < 2:
            filtered = header_words[:4]
        
        spacings = [filtered[i + 1][0] - filtered[i][0] for i in range(len(filtered) - 1)]
        col_width = sum(spacings) / len(spacings) if spacings else 150.0
        
        cols = [(hw[1], hw[0] / fx) for hw in filtered]
        
        if self.verbose:
            print(f"🔍 UIDetector: Bidding columns: {cols}, width={col_width / fx:.0f}px")
        
        return cols, col_width / fx

    def detect_hand_card_peaks(self, hand_img, min_peaks=4):
        """
        Detect individual card positions in the player hand strip using
        a color profile projection. Works at any resolution.
        
        Returns list of dicts: [{x_center, color}, ...]
        """
        h_strip = hand_img.shape[0]
        w_strip = hand_img.shape[1]
        
        # Dynamically find the card strip vertically within hand_img to handle tall/calibrated crops
        hsv_full = cv2.cvtColor(hand_img, cv2.COLOR_BGR2HSV)
        card_mask = (hsv_full[:, :, 1] < 50) & (hsv_full[:, :, 2] > 170)
        row_card_counts = np.sum(card_mask, axis=1)
        card_rows = np.where(row_card_counts > 0.05 * w_strip)[0]
        
        if len(card_rows) >= 10:
            y_start = card_rows[0]
            y_end = card_rows[-1]
            y_start = max(0, y_start - 2)
            y_end = min(h_strip - 1, y_end + 2)
            hand_img = hand_img[y_start:y_end+1, :]
            h_strip = hand_img.shape[0]
            
        target_h = 60
        scale = target_h / h_strip if h_strip != target_h else 1.0
        if scale != 1.0:
            hand_img = cv2.resize(hand_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        hsv = cv2.cvtColor(hand_img, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 40, 40])
        upper_red1 = np.array([25, 255, 255])
        lower_red2 = np.array([165, 40, 40])
        upper_red2 = np.array([180, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
        
        mask_black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 60, 100]))
        
        mask_suit = mask_red + mask_black
        
        suit_row_start = int(target_h * 0.68)
        suit_row_end = int(target_h * 0.90)
        profile = np.sum(mask_suit[suit_row_start:suit_row_end, :] > 0, axis=0).astype(np.float32)
        
        kernel_size = max(7, int(w_strip * 0.015) | 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones(kernel_size) / float(kernel_size)
        smoothed = np.convolve(profile, kernel, mode='same')
        
        min_dist = 15
        
        peaks = []
        for x in range(min_dist, len(smoothed) - min_dist):
            val = smoothed[x]
            if val >= 1.5:
                is_max = True
                for dx in range(-min_dist, min_dist + 1):
                    if smoothed[x + dx] > val:
                        is_max = False
                        break
                if is_max:
                    if not peaks or (x - peaks[-1]["x_suit"]) >= min_dist:
                        col_red = np.sum(mask_red[suit_row_start:suit_row_end, x] > 0)
                        col_black = np.sum(mask_black[suit_row_start:suit_row_end, x] > 0)
                        color = "RED" if col_red >= col_black else "BLACK"
                        peaks.append({"x_suit": x, "color": color})
        
        if self.verbose:
            print(f"🔍 UIDetector: Hand peaks: {len(peaks)} cards at {[p['x_suit'] for p in peaks]}")
        
        return peaks, scale

    def auto_bootstrap_templates(self, hand_img, output_dir="templates"):
        """
        Auto-detect suit icon positions in the hand image and extract templates.
        Works at any resolution by finding distinct suit colors and positions.
        """
        # Crop and resize hand_img to normalized height 60 first to ensure proper template sizes
        h_strip = hand_img.shape[0]
        w_strip = hand_img.shape[1]
        
        hsv_full = cv2.cvtColor(hand_img, cv2.COLOR_BGR2HSV)
        card_mask = (hsv_full[:, :, 1] < 50) & (hsv_full[:, :, 2] > 170)
        row_card_counts = np.sum(card_mask, axis=1)
        card_rows = np.where(row_card_counts > 0.05 * w_strip)[0]
        
        if len(card_rows) >= 10:
            y_start = card_rows[0]
            y_end = card_rows[-1]
            y_start = max(0, y_start - 2)
            y_end = min(h_strip - 1, y_end + 2)
            hand_img = hand_img[y_start:y_end+1, :]
            h_strip = hand_img.shape[0]
            
        if h_strip != 60:
            scale = 60.0 / h_strip
            hand_img = cv2.resize(hand_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            
        peaks, scale = self.detect_hand_card_peaks(hand_img, min_peaks=1)
        
        if len(peaks) < 4:
            if self.verbose:
                print(f"⚠️ UIDetector: Found {len(peaks)} peaks, need at least 4 for template bootstrap.")
            return False
        
        color_peaks = {"RED": [], "BLACK": []}
        for p in peaks:
            color_peaks[p["color"]].append(p["x_suit"])
        
        red_suits = []
        black_suits = []
        
        if len(color_peaks["RED"]) >= 2:
            red_suits = sorted(color_peaks["RED"])
        if len(color_peaks["BLACK"]) >= 2:
            black_suits = sorted(color_peaks["BLACK"])
        
        suit_names = []
        if len(black_suits) >= 2:
            suit_names.extend(["spade", "club"])
        if len(red_suits) >= 2:
            suit_names.extend(["heart", "diamond"])
        
        if len(suit_names) < 4:
            if self.verbose:
                print(f"⚠️ UIDetector: Could not identify all 4 suits from {len(peaks)} peaks.")
            return False
        
        os.makedirs(output_dir, exist_ok=True)
        
        h_60 = 60
        suit_y_start = int(h_60 * 0.68)
        suit_y_end = int(h_60 * 0.90)
        crop_size = 13
        
        all_positions = sorted(
            [(black_suits[0], "spade"), (black_suits[-1], "club"),
             (red_suits[0], "heart"), (red_suits[-1], "diamond")],
            key=lambda x: x[0]
        )
        
        for x_center, suit_name in all_positions:
            x_start = max(0, x_center - crop_size // 2)
            x_end = x_start + crop_size
            
            crop = hand_img[suit_y_start:suit_y_end, x_start:x_end]
            if crop.size == 0:
                continue
            
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            out_path = os.path.join(output_dir, f"{suit_name}.png")
            cv2.imwrite(out_path, gray_crop)
            if self.verbose:
                print(f"   Saved {out_path} (center={x_center})")
        
        return True

    def detect_vulnerability(self, ui_img, fx=2.0):
        """
        Detect vulnerability indicators (red/green dots or text) near player directions.
        Returns dict like {"N": "none", "E": "none", "S": "vul", "W": "vul"} or None.
        """
        hsv = cv2.cvtColor(ui_img, cv2.COLOR_BGR2HSV)
        
        lower_vul = np.array([0, 100, 100])
        upper_vul = np.array([10, 255, 255])
        lower_vul2 = np.array([160, 100, 100])
        upper_vul2 = np.array([180, 255, 255])
        vul_mask = cv2.inRange(hsv, lower_vul, upper_vul) + cv2.inRange(hsv, lower_vul2, upper_vul2)
        
        lower_safe = np.array([35, 100, 100])
        upper_safe = np.array([85, 255, 255])
        safe_mask = cv2.inRange(hsv, lower_safe, upper_safe)
        
        h, w = ui_img.shape[:2]
        
        quadrants = {
            "S": (w // 4, int(h * 0.7), w // 2, h),
            "W": (0, h // 4, w // 4, int(h * 0.75)),
            "N": (w // 4, 0, int(w * 0.75), h // 4),
            "E": (int(w * 0.75), h // 4, w, int(h * 0.75)),
        }
        
        vul_result = {}
        for direction, (x1, y1, x2, y2) in quadrants.items():
            region_vul = vul_mask[y1:y2, x1:x2]
            region_safe = safe_mask[y1:y2, x1:x2]
            
            vul_area = np.sum(region_vul > 0)
            safe_area = np.sum(region_safe > 0)
            
            if vul_area > safe_area * 2 and vul_area > 50:
                vul_result[direction] = "vul"
            else:
                vul_result[direction] = "none"
        
        return vul_result if any(v == "vul" for v in vul_result.values()) else None

    def detect_dealer_indicator(self, bidding_img, fx=4.0):
        """
        Detect the dealer indicator (small marker near one of the N/E/S/W headers).
        Returns the dealer direction or None.
        """
        gray = cv2.cvtColor(bidding_img, cv2.COLOR_BGR2GRAY)
        scaled = cv2.resize(gray, (0, 0), fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
        
        _, thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        img_h, img_w = scaled.shape[:2]
        
        direction_regions = {
            "W": (0, 0, img_w // 4, img_h // 3),
            "N": (img_w // 4, 0, int(img_w * 0.75), img_h // 3),
            "E": (int(img_w * 0.75), 0, img_w, img_h // 3),
            "S": (img_w // 4, img_h // 3, int(img_w * 0.75), 2 * img_h // 3),
        }
        
        direction_scores = {}
        for direction, (rx1, ry1, rx2, ry2) in direction_regions.items():
            score = 0
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                cx, cy = x + w // 2, y + h // 2
                if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                    if w < 20 and h < 20:
                        score += 1
            direction_scores[direction] = score
        
        if direction_scores:
            best = max(direction_scores, key=direction_scores.get)
            if direction_scores[best] > 0:
                return best
        
        return None

    def find_suit_button_positions(self, ui_img, ui_roi, nt_center_x, nt_center_y, fx=2.0):
        """
        Dynamically locate suit buttons relative to the NT button position.
        Returns dict of suit -> (x, y) screen coordinates.
        """
        if not ui_roi:
            ui_roi = {"x": 0, "y": 0, "width": ui_img.shape[1], "height": ui_img.shape[0]}

        nt_rel_x = nt_center_x - ui_roi["x"]
        nt_rel_y = nt_center_y - ui_roi["y"]
        
        ui_width = ui_roi.get("width", 1200)
        base_spacing = 50.0 * (ui_width / 1200.0)
        
        # The 4 suit buttons (S, H, D, C) are to the left of NT.
        # Let's set search boundaries relative to ui_img.
        target_region_w = int(base_spacing * 5.5)
        target_region_h = int(base_spacing * 1.5)
        
        search_x1 = max(0, int(nt_rel_x - target_region_w + base_spacing * 0.75))
        search_y1 = max(0, int(nt_rel_y - target_region_h // 2))
        search_x2 = min(ui_img.shape[1], int(nt_rel_x + base_spacing * 0.75))
        search_y2 = min(ui_img.shape[0], int(nt_rel_y + target_region_h // 2))
        
        region = ui_img[search_y1:search_y2, search_x1:search_x2]
        
        # Convert search region to grayscale.
        region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Blur and run Canny edge detection.
        blurred = cv2.GaussianBlur(region_gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 30, 100)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Expected button dimensions
        expected_btn_w = base_spacing * 0.75
        expected_btn_h = base_spacing * 0.6
        
        min_w, max_w = expected_btn_w * 0.5, expected_btn_w * 2.0
        min_h, max_h = expected_btn_h * 0.5, expected_btn_h * 2.0
        
        candidates = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if min_w <= w <= max_w and min_h <= h <= max_h:
                cx = search_x1 + x + w // 2
                cy = search_y1 + y + h // 2
                if abs(cy - nt_rel_y) < base_spacing * 0.5:
                    candidates.append((cx, cy))
                    
        # Remove duplicates/very close candidates
        unique_candidates = []
        for cx, cy in candidates:
            if not any(abs(cx - ux) < base_spacing * 0.3 for ux, uy in unique_candidates):
                unique_candidates.append((cx, cy))
                
        # Fit suit grid from NT button to the left
        # Suit order: C (leftmost), D, H, S, NT (rightmost)
        suits = ["NT", "S", "H", "D", "C"]
        result = {
            "NT": (nt_center_x, nt_center_y)
        }
        
        for k in range(1, 5):
            expected_x = nt_rel_x - k * base_spacing
            
            # Find candidate closest to expected_x
            best_cand = None
            min_dist = base_spacing * 0.4
            for cx, cy in unique_candidates:
                dist = abs(cx - expected_x)
                if dist < min_dist:
                    min_dist = dist
                    best_cand = (cx, cy)
                    
            if best_cand:
                rx, ry = best_cand
            else:
                rx, ry = expected_x, nt_rel_y
                
            result[suits[k]] = (int(rx + ui_roi["x"]), int(ry + ui_roi["y"]))
            
        if self.verbose:
            print(f"🔍 UIDetector: Found suit button positions: {result}")
            
        return result

    def extract_contract_from_bids(self, bids):
        """
        Extract the final contract, declarer, and doubler from bid history.
        Returns dict with keys: contract, level, suit, doubled, declarer.
        """
        if not bids:
            return None
        
        last_contract_bid = None
        last_contract_idx = -1
        
        for i, (direction, bid) in enumerate(bids):
            bid_upper = bid.upper()
            if bid_upper not in ("PASS", "DBL", "RDBL"):
                last_contract_bid = bid
                last_contract_idx = i
        
        if last_contract_bid is None or last_contract_idx < 0:
            return None
        
        level = int(last_contract_bid[0])
        suit_text = last_contract_bid[1:]
        suit_map = {"S": "spade", "H": "heart", "D": "diamond", "C": "club", "NT": "notrump"}
        suit = suit_map.get(suit_text, suit_text)
        
        doubled = "none"
        for direction, bid in bids[last_contract_idx + 1:]:
            bid_upper = bid.upper()
            if bid_upper == "DBL":
                doubled = "doubled"
            elif bid_upper == "RDBL":
                doubled = "redoubled"
        
        declarer = bids[last_contract_idx][0]
        
        return {
            "contract": f"{level}{suit_text}",
            "level": level,
            "suit": suit,
            "suit_symbol": suit_text,
            "doubled": doubled,
            "declarer": declarer,
        }

    def track_trick_winner(self, completed_trick, trump_suit=None):
        """
        Determine the winner of a completed trick.
        Returns the winning direction.
        """
        if not completed_trick:
            return None
        
        card_played = []
        for direction in ["N", "E", "S", "W"]:
            card = completed_trick.get(direction)
            if card:
                card_played.append((direction, card))
        
        if not card_played:
            return None
        
        suit_order = {"S": 3, "H": 2, "D": 1, "C": 0}
        rank_order = {"A": 14, "K": 13, "Q": 12, "J": 11, "T": 10,
                      "9": 9, "8": 8, "7": 7, "6": 6, "5": 5, "4": 4, "3": 3, "2": 2}
        
        led_direction, led_card = card_played[0]
        led_suit = led_card[0]
        
        best_direction = led_direction
        best_card = led_card
        
        for direction, card in card_played[1:]:
            card_suit = card[0]
            card_rank = card[1:]
            
            if trump_suit and card_suit == trump_suit:
                if best_card[0] != trump_suit or rank_order.get(card_rank, 0) > rank_order.get(best_card[1:], 0):
                    best_direction = direction
                    best_card = card
            elif card_suit == led_suit and best_card[0] == led_suit:
                if rank_order.get(card_rank, 0) > rank_order.get(best_card[1:], 0):
                    best_direction = direction
                    best_card = card
        
        return best_direction

    def get_remaining_suit_counts(self, initial_hand, played_cards):
        """
        Calculate remaining cards per suit in South's hand.
        played_cards: list of PBN card strings like "SA", "HK", etc.
        """
        suit_map = {"S": "spade", "H": "heart", "D": "diamond", "C": "club"}
        
        remaining = {"spade": 0, "heart": 0, "diamond": 0, "club": 0}
        
        for card in initial_hand:
            suit = card.get("suit")
            if suit in remaining:
                remaining[suit] += 1
        
        for pbn_card in played_cards:
            if len(pbn_card) >= 2:
                suit_letter = pbn_card[0]
                suit = suit_map.get(suit_letter)
                if suit and remaining.get(suit, 0) > 0:
                    remaining[suit] -= 1
        
        return remaining


def col_width_from_image(img):
    """Fallback column width estimate from image width."""
    return img.shape[1] / 4
