#!/usr/bin/env python3
"""
Card & Bid Analyzer for Bridge Bot.
Uses OpenCV and PyTesseract to extract information from captured screen regions.
"""

import os
import re
import cv2
import numpy as np
import pytesseract
from PIL import Image

class BridgeAnalyzer:
    def __init__(self, templates_dir="templates", verbose=False):
        self.templates_dir = templates_dir
        self.suit_templates = {}
        self.verbose = verbose
        self.load_templates()

    def load_templates(self):
        """Loads suit templates from the templates directory if they exist."""
        if not os.path.exists(self.templates_dir):
            return
        
        for suit in ["spade", "heart", "diamond", "club"]:
            path = os.path.join(self.templates_dir, f"{suit}.png")
            if os.path.exists(path):
                # Read as grayscale
                self.suit_templates[suit] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        if self.suit_templates and self.verbose:
            print(f"Loaded {len(self.suit_templates)} suit templates.")

    def preprocess_for_ocr(self, img, fx=4.0, thresh_val=127):
        """Preprocesses image for Tesseract OCR (grayscale, scale up, threshold)."""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Scale up to improve OCR accuracy on small text
        scaled = cv2.resize(gray, (0, 0), fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
        
        # Simple thresholding, Otsu, or no threshold
        if thresh_val == "otsu":
            thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        elif thresh_val is not None:
            thresh = cv2.threshold(scaled, thresh_val, 255, cv2.THRESH_BINARY)[1]
        else:
            thresh = scaled
        
        return thresh

    def clean_header_text(self, text):
        # Remove non-alphabetic characters
        cleaned = re.sub(r'[^a-zA-Z]', '', text).upper()
        if not cleaned:
            return None
            
        # Check if the cleaned word matches or starts with one of the headers
        if "SOUTH" in cleaned or cleaned.startswith("S"):
            return "S"
        if "WEST" in cleaned or cleaned.startswith("W"):
            return "W"
        if "NORTH" in cleaned or cleaned.startswith("N"):
            return "N"
        if "EAST" in cleaned or cleaned.startswith("E"):
            return "E"
            
        return None

    def standardize_bid(self, bid_str):
        b_clean = bid_str.upper().replace(" ", "")

        # Normalize common suit-symbol OCR artifacts to letter suits.
        # Example: '1@' is often an OCR form of '1♣'.
        symbol_map = {
            "♠": "S",
            "♥": "H",
            "♦": "D",
            "♣": "C",
            "@": "S",
            "&": "C",
        }
        for sym, repl in symbol_map.items():
            b_clean = b_clean.replace(sym, repl)
        
        # Replace common digit 1 typos if followed by suit
        if len(b_clean) >= 2 and b_clean[0] in ["I", "L", "T", "!", "|"]:
            suit_part = b_clean[1:]
            if suit_part in ["NT", "N", "S", "H", "D", "C", "SPADES", "HEARTS", "DIAMONDS", "CLUBS"]:
                b_clean = "1" + suit_part
                
        if b_clean in ["PAS", "PA", "PASSED"]:
            return "PASS"
        elif b_clean in ["DBL", "DOUBLE", "X"]:
            return "DBL"
        elif b_clean in ["RDBL", "REDOUBLE", "XX"]:
            return "RDBL"
        elif len(b_clean) >= 2 and b_clean[0].isdigit():
            digit = b_clean[0]
            suit = b_clean[1:]
            if suit in ["SPADES", "S"]:
                suit = "S"
            elif suit in ["HEARTS", "H"]:
                suit = "H"
            elif suit in ["DIAMONDS", "D"]:
                suit = "D"
            elif suit in ["CLUBS", "C"]:
                suit = "C"
            elif suit in ["N", "NT"]:
                suit = "NT"
            return f"{digit}{suit}"
        return b_clean

    def extract_bids(self, bidding_img):
        """
        Runs OCR on the bidding history image and parses the bids using spatial column alignment.
        Returns a list of tuples (direction, cleaned_bid), e.g. [('N', 'PASS'), ('E', 'PASS')].
        """
        import csv
        from io import StringIO
        
        # Use scaled grayscale image without thresholding to preserve low-contrast red headers
        processed = self.preprocess_for_ocr(bidding_img, thresh_val=None)
        
        # Run OCR with image_to_data
        data_str = pytesseract.image_to_data(processed, config="--psm 6", output_type=pytesseract.Output.STRING)
        
        f = StringIO(data_str)
        reader = csv.reader(f, delimiter='\t')
        
        try:
            header = next(reader)
        except StopIteration:
            return []
            
        try:
            left_idx = header.index('left')
            top_idx = header.index('top')
            width_idx = header.index('width')
            height_idx = header.index('height')
            text_idx = header.index('text')
        except ValueError:
            return []
            
        words = []
        for row in reader:
            if len(row) <= text_idx:
                continue
            text = row[text_idx].strip()
            if not text:
                continue
            words.append({
                "left": int(row[left_idx]),
                "top": int(row[top_idx]),
                "width": int(row[width_idx]),
                "height": int(row[height_idx]),
                "text": text,
                "center_x": int(row[left_idx]) + int(row[width_idx]) // 2
            })
            
        if not words:
            return []
            
        # 1. Extract headers dynamically
        all_header_candidates = []
        for w in words:
            dir_key = self.clean_header_text(w["text"])
            # Header row is in the top 35% of the height
            if dir_key and w["top"] < processed.shape[0] * 0.35:
                all_header_candidates.append(w)
                
        if not all_header_candidates:
            if self.verbose:
                print("⚠️ No direction headers detected in bidding area.")
            return []
            
        # Find the topmost row of direction headers (minimum top coordinate)
        min_top = min(w["top"] for w in all_header_candidates)
        
        # Group header words in the same row
        header_words = []
        for w in all_header_candidates:
            if w["top"] - min_top < 30:  # 30 scaled pixels row clustering
                header_words.append((w["center_x"], self.clean_header_text(w["text"]), w["top"]))
                
        header_words.sort()  # Sort by center_x
        
        # Define the fixed column width (150 scaled pixels on IntoBridge docked UI)
        col_width = 150.0
        
        # We find the best alignment that fits the detected headers.
        # Possible rotation orders:
        full_order = ["N", "E", "S", "W"]
        
        # We will test candidate values for c0 (center of Col 0).
        # Since the columns must lie within the image width, and Col 0 is leftmost,
        # its center c0 is typically close to min_cx, or min_cx - k*col_width.
        min_cx = header_words[0][0]
        
        best_rotation = ["N", "E", "S", "W"]
        best_c0 = min_cx
        max_matches = -1
        
        # We check all possible slots (0 to 3) for the leftmost detected header
        for leftmost_slot in range(4):
            candidate_c0 = min_cx - (leftmost_slot * col_width)
            
            # Now we check all 4 rotations of compass headers
            for r in range(4):
                candidate_rotation = full_order[r:] + full_order[:r]
                
                # Calculate matches
                matches = 0
                for center_x, direction, top in header_words:
                    slot_idx = int(round((center_x - candidate_c0) / col_width))
                    if 0 <= slot_idx < 4:
                        if candidate_rotation[slot_idx] == direction:
                            matches += 1
                            
                if matches > max_matches:
                    max_matches = matches
                    best_rotation = candidate_rotation
                    best_c0 = candidate_c0
                    
        # Reconstruct the 4 column centers and directions based on the winning combination
        col_centers = [best_c0 + idx * col_width for idx in range(4)]
        col_dirs = best_rotation
        max_header_top = max(hw[2] for hw in header_words)
        
        if self.verbose:
            print(f"Bidding columns reconstructed: {list(zip(col_dirs, col_centers))}")
            
        # 2. Extract bid words (below headers)
        bid_words = []
        for w in words:
            if w["top"] > max_header_top + 10:
                # Map to closest column
                closest_idx = 0
                min_dist = float('inf')
                for idx, center in enumerate(col_centers):
                    dist = abs(w["center_x"] - center)
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = idx
                
                w["col_idx"] = closest_idx
                w["direction"] = col_dirs[closest_idx]
                bid_words.append(w)
                
        # Group bid words into rows based on top coordinate
        bid_words.sort(key=lambda w: w["top"])
        
        word_rows = []
        if bid_words:
            current_row = [bid_words[0]]
            for w in bid_words[1:]:
                if w["top"] - current_row[-1]["top"] < 30:
                    current_row.append(w)
                else:
                    word_rows.append(current_row)
                    current_row = [w]
            word_rows.append(current_row)
            
        # Regex to validate standard bids
        bid_pattern = re.compile(
            r'^(PASS|PAS|PA|PASSED|DBL|DOUBLE|RDBL|REDOUBLE|X|XX|[1-7]\s*(?:NT|N|S|H|D|C|SPADES|HEARTS|DIAMONDS|CLUBS))$',
            re.IGNORECASE
        )
        
        direction_bids = []
        for row in word_rows:
            col_groups = {}
            for w in row:
                col_groups.setdefault(w["col_idx"], []).append(w)
                
            row_bids = []
            for col_idx in sorted(col_groups.keys()):
                g_words = col_groups[col_idx]
                g_words.sort(key=lambda w: w["left"])
                combined_text = " ".join(w["text"] for w in g_words)
                std_text = self.standardize_bid(combined_text)
                
                if bid_pattern.match(std_text):
                    row_bids.append((col_idx, col_dirs[col_idx], std_text))
                elif self.verbose:
                    print(f"Skipping non-bid text: '{combined_text}' -> '{std_text}'")
                    
            row_bids.sort()
            for col_idx, direction, std_text in row_bids:
                direction_bids.append((direction, std_text))
                
        return direction_bids

    def detect_bidding_headers(self, bidding_img):
        """
        Parses the bidding image to find the column headers (N, E, S, W).
        Returns a list of 4 header strings, e.g. ["W", "N", "E", "S"].
        """
        processed = self.preprocess_for_ocr(bidding_img)
        text = pytesseract.image_to_string(processed, config="--psm 6")
        lines = text.split("\n")
        
        # Look for headers in the first few lines
        header_patterns = [
            ("W", ["WEST", "W"]),
            ("N", ["NORTH", "N"]),
            ("E", ["EAST", "E"]),
            ("S", ["SOUTH", "S"])
        ]
        
        for line in lines[:3]:
            line_upper = line.upper()
            positions = []
            for key, names in header_patterns:
                for name in names:
                    idx = line_upper.find(name)
                    if idx != -1:
                        positions.append((idx, key))
                        break
            if len(positions) >= 2:
                positions.sort()
                ordered_keys = [key for _, key in positions]
                full_order = ["W", "N", "E", "S"]
                first_key = ordered_keys[0]
                idx = full_order.index(first_key)
                rotated = full_order[idx:] + full_order[:idx]
                return rotated
                
        return ["W", "N", "E", "S"]

    def classify_suit_by_color_shape(self, suit_img):
        """
        Fallback classification of suit based on color and geometric shapes.
        Works well on standard bright UI elements.
        """
        # Convert to HSV to analyze color
        hsv = cv2.cvtColor(suit_img, cv2.COLOR_BGR2HSV)
        
        # Define red/orange color mask (Hearts/Diamonds)
        # Red and orange spans 0-25 and 170-180 in OpenCV Hue
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([25, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        
        # Calculate percentage of red pixels
        red_ratio = np.sum(red_mask > 0) / (suit_img.shape[0] * suit_img.shape[1])
        
        # Grayscale and threshold for shape contours
        gray = cv2.cvtColor(suit_img, cv2.COLOR_BGR2GRAY)
        # Assuming light background or dark background - do adaptive binary
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        
        # If it's mostly white on dark background, we may need to invert
        # We want the shape to be white (255) and background black (0)
        border_mean = (np.mean(thresh[0, :]) + np.mean(thresh[-1, :]) + 
                       np.mean(thresh[:, 0]) + np.mean(thresh[:, -1])) / 4
        if border_mean > 127:
            thresh = cv2.bitwise_not(thresh)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return "unknown"
            
        # Get largest contour
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        
        if area < 10:
            return "unknown"
            
        perimeter = cv2.arcLength(c, True)
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / h
        
        # Convex Hull and Solidity
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        
        # Circularity
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        # Classify based on color & shape metrics
        # Classify based on color & shape metrics
        is_red = red_ratio > 0.015
        
        # Print shape metrics for debugging
        # print(f"DEBUG SUIT SHAPE: area={area:.1f}, solidity={solidity:.3f}, circularity={circularity:.3f}, aspect_ratio={aspect_ratio:.2f}, is_red={is_red}")
        
        if is_red:
            # Diamond vs Heart
            # Diamonds have high solidity (almost a diamond/square) and 4 sharp corners.
            # Hearts have a notch at the top, lower solidity.
            if solidity > 0.85:
                return "diamond"
            else:
                return "heart"
        else:
            # Spade vs Club
            # Spades have a sharp point at the top and high solidity.
            # Clubs have 3 distinct lobes, very low solidity/circularity.
            if solidity > 0.78:
                return "spade"
            else:
                return "club"

    def classify_suit_template_matching(self, suit_img):
        """
        Classify suit using cv2.matchTemplate against loaded templates.
        Pre-filters candidate templates using color (red vs black) to eliminate cross-color errors.
        """
        if not self.suit_templates:
            return None
            
        # Determine color channel first (Red vs Black)
        hsv = cv2.cvtColor(suit_img, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([25, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        
        red_ratio = np.sum(red_mask > 0) / (suit_img.shape[0] * suit_img.shape[1])
        is_red = red_ratio > 0.015
        
        # Route to allowed suit templates based on color pigment
        allowed_suits = ["heart", "diamond"] if is_red else ["spade", "club"]
        
        gray = cv2.cvtColor(suit_img, cv2.COLOR_BGR2GRAY)
        best_match = None
        best_score = -1.0
        
        for suit in allowed_suits:
            template = self.suit_templates.get(suit)
            if template is None:
                continue
                
            t_h, t_w = template.shape[:2]
            g_h, g_w = gray.shape[:2]
            
            # Slide matching window
            if g_h < t_h or g_w < t_w:
                gray_search = cv2.resize(gray, (max(g_w, t_w), max(g_h, t_h)))
            else:
                gray_search = gray
                
            res = cv2.matchTemplate(gray_search, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > best_score:
                best_score = max_val
                best_match = suit
            
        if self.verbose:
            print(f"Debug Match: {best_match} (score: {best_score:.3f}, is_red: {is_red})")
        if best_score > 0.70:
            return best_match
        return None

    def extract_card(self, card_img):
        """
        Parses a single card image crop.
        Returns a tuple (rank, suit) or (None, None).
        """
        h, w = card_img.shape[:2]
        if w < 10 or h < 10:
            return None, None
            
        # Tightened crops skipping the left border (IntoBridge UI)
        # Ranks: top-left corner (top 43%, left 5 to 45%)
        # Suit: search region slightly larger for sliding template search (top 40% to 93%, left 2 to 50%)
        if h == 60:
            rank_crop = card_img[9:35, 4:18]
        else:
            rank_crop = card_img[2:int(h*0.43), 5:int(w*0.45)]
        suit_crop = card_img[int(h*0.40):int(h*0.93), 2:int(w*0.50)]
        
        if rank_crop.size == 0 or suit_crop.size == 0:
            return None, None
            
        # Extract Rank
        processed_rank = self.preprocess_for_ocr(rank_crop)

        def normalize_rank_text(raw_text):
            rank_text = raw_text.strip().upper().replace(" ", "")
            if not rank_text:
                return None

            # Tesseract often reads 10 as 1.
            if "10" in rank_text:
                return "10"
            if rank_text == "1":
                return "10"

            # Common Queen misreads from curved glyphs.
            if rank_text in ["0", "O", "D"]:
                return "Q"

            valid_ranks = ["A", "K", "Q", "J", "T", "10", "9", "8", "7", "6", "5", "4", "3", "2"]
            return rank_text if rank_text in valid_ranks else None

        # First pass with single-word OCR, then fallback to single-char OCR.
        rank_text = None
        for psm in [8, 10]:
            custom_config = f"--psm {psm} -c tessedit_char_whitelist=AKQJT1098765432"
            raw_rank = pytesseract.image_to_string(processed_rank, config=custom_config)
            rank_text = normalize_rank_text(raw_rank)
            if rank_text:
                break
            
        # Extract Suit
        suit = None
        if self.suit_templates:
            suit = self.classify_suit_template_matching(suit_crop)
            
        if not suit:
            suit = self.classify_suit_by_color_shape(suit_crop)
            
        return rank_text, suit

    def extract_multiple_cards(self, cards_img, expected_count=4):
        """
        Finds and extracts multiple cards from an image containing several cards.
        Usually useful for parsing the trick play area or the hand.
        """
        # Grayscale and threshold
        gray = cv2.cvtColor(cards_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_cards = []
        for c in contours:
            area = cv2.contourArea(c)
            # Filter contours by size to match playing card shape
            if area < 1000:
                continue
                
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w)/h
            # Standard cards aspect ratio is around 0.7, but could be tilted/partially overlapping
            if aspect_ratio < 0.3 or aspect_ratio > 1.8:
                continue
                
            card_crop = cards_img[y:y+h, x:x+w]
            rank, suit = self.extract_card(card_crop)
            # Filter out any detection that lacks a recognized rank 
            # (e.g. face-down card backs or background elements)
            if rank and suit:
                detected_cards.append({
                    "rank": rank,
                    "suit": suit,
                    "bbox": {"x": x, "y": y, "w": w, "h": h}
                })
                
        return detected_cards

    def extract_hand_cards_linear(self, hand_img):
        """Fallback linear card slicing method."""
        gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        col_sums = np.sum(thresh, axis=0)
        card_cols = np.where(col_sums > 1000)[0]
        
        if len(card_cols) < 30:
            return []
            
        x_start = card_cols[0]
        x_end = card_cols[-1]
        w_strip = x_end - x_start
        h_strip = hand_img.shape[0]
        
        num_cards = int(round((w_strip - 40) / 26.0) + 1)
        num_cards = max(1, min(13, num_cards))
        
        if num_cards > 1:
            step = (w_strip - 40) / (num_cards - 1)
        else:
            step = 0
            
        detected_cards = []
        for i in range(num_cards):
            x_card = int(x_start + i * step)
            card_crop = hand_img[0:h_strip, x_card:min(x_card + 40, hand_img.shape[1])]
            rank, suit = self.extract_card(card_crop)
            detected_cards.append({
                "rank": rank,
                "suit": suit,
                "bbox": {"x": x_card, "y": 0, "w": 40, "h": h_strip}
            })
        return detected_cards

    def extract_hand_cards(self, hand_img):
        """
        Extracts cards from a player hand row crop.
        Finds individual cards by sliding a suit template matcher across the horizontal row,
        detecting the precise peaks of the suit symbol matches.
        Handles suit gaps and variable card spacing perfectly.
        """
        if not self.suit_templates:
            print("⚠️ No suit templates loaded. Falling back to linear slicing.")
            return self.extract_hand_cards_linear(hand_img)
            
        h_strip = hand_img.shape[0]
        w_strip = hand_img.shape[1]
        
        # Define HSV color boundaries for the sliding loop
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([25, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        # 1. Slide window across the width of the card row
        scores = []
        for x in range(0, w_strip - 15):
            # Crop a potential suit symbol area (y from 37 to 50 for 60px height to match 13x13 templates)
            y_start = 37 if h_strip == 60 else int(h_strip*0.43)
            y_end = 50 if h_strip == 60 else int(h_strip*0.90)
            suit_crop = hand_img[y_start:y_end, x:x+13]
            
            # Determine color channel (Red vs Black)
            hsv = cv2.cvtColor(suit_crop, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = mask1 + mask2
            red_ratio = np.sum(red_mask > 0) / (suit_crop.shape[0] * suit_crop.shape[1])
            is_red = red_ratio > 0.015
            
            if x == 102 and self.verbose:
                print(f"DEBUG x=102: red_ratio={red_ratio:.4f}, is_red={is_red}")
            
            allowed_suits = ["heart", "diamond"] if is_red else ["spade", "club"]
            gray_crop = cv2.cvtColor(suit_crop, cv2.COLOR_BGR2GRAY)
            
            # Find best match score among allowed color-routed suits
            best_score = -1.0
            best_suit = None
            for suit in allowed_suits:
                template = self.suit_templates.get(suit)
                if template is not None:
                    t_h, t_w = template.shape[:2]
                    if gray_crop.shape[0] == t_h and gray_crop.shape[1] == t_w:
                        res = cv2.matchTemplate(gray_crop, template, cv2.TM_CCOEFF_NORMED)
                        max_val = res[0][0]
                        if max_val > best_score:
                            best_score = max_val
                            best_suit = suit
                        
            scores.append((x, best_score, best_suit))
            
        # 2. Find local peaks (local maxima) in the scores
        peaks = []
        min_distance = 13  # Minimum distance in pixels between card suit symbols (cards overlap up to 15px)
        
        for idx in range(1, len(scores) - 1):
            x, score, suit = scores[idx]
            
            # Lowered threshold to 0.60 to ensure all symbols are detected
            if score > 0.60:
                # Check if it's a local maximum
                is_peak = True
                # Check neighbors in a radius of min_distance
                start_check = max(0, idx - min_distance)
                end_check = min(len(scores), idx + min_distance)
                for j in range(start_check, end_check):
                    if scores[j][1] > score:
                        is_peak = False
                        break
                        
                if is_peak:
                    # Prevent duplicate peaks at flat tops
                    if not peaks or (x - peaks[-1]["x_suit"]) >= min_distance:
                        peaks.append({
                            "x_suit": x,
                            "score": score,
                            "suit": suit
                        })
                        
        # If we didn't find enough cards, fall back to linear slicing
        if len(peaks) < 4:
            if self.verbose:
                print(f"⚠️ Template peak detection found too few cards ({len(peaks)}). Falling back to linear slicing.")
            return self.extract_hand_cards_linear(hand_img)
            
        if self.verbose:
            print(f"Detected {len(peaks)} card suit peaks: {[(p['x_suit'], p['suit'], round(p['score'], 2)) for p in peaks]}")
        
        detected_cards = []
        for p in peaks:
            # Card left edge is approximately 5 pixels to the left of the suit symbol
            x_card = max(0, p["x_suit"] - 5)
            card_crop = hand_img[0:h_strip, x_card:min(x_card + 40, w_strip)]
            
            rank, suit = self.extract_card(card_crop)
            detected_cards.append({
                "rank": rank,
                "suit": suit or p["suit"],  # Use validated suit, fall back to peak suit
                "bbox": {"x": x_card, "y": 0, "w": 40, "h": h_strip}
            })
            
        return detected_cards

if __name__ == "__main__":
    # Test Analyzer interface
    print("Bridge Play UI Analyzer initialized successfully.")
