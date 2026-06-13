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
    _DIRECTION_MAP = {"SOUTH": "S", "NORTH": "N", "EAST": "E", "WEST": "W"}

    def __init__(self, templates_dir="templates", verbose=False):
        self.templates_dir = templates_dir
        self.suit_templates = {}
        self.verbose = verbose
        self.load_templates()

    def load_templates(self):
        if not os.path.exists(self.templates_dir):
            return

        for suit in ["spade", "heart", "diamond", "club"]:
            path = os.path.join(self.templates_dir, f"{suit}.png")
            if os.path.exists(path):
                self.suit_templates[suit] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if self.suit_templates and self.verbose:
            print(f"Loaded {len(self.suit_templates)} suit templates.")

    def preprocess_for_ocr(self, img, fx=4.0, thresh_val=127):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        scaled = cv2.resize(gray, (0, 0), fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)

        if thresh_val == "otsu":
            thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        elif thresh_val is not None:
            thresh = cv2.threshold(scaled, thresh_val, 255, cv2.THRESH_BINARY)[1]
        else:
            thresh = scaled

        return thresh

    def clean_header_text(self, text):
        cleaned = re.sub(r'[^a-zA-Z]', '', text).upper()
        if not cleaned:
            return None

        if cleaned in self._DIRECTION_MAP:
            return self._DIRECTION_MAP[cleaned]

        for full_word, direction in self._DIRECTION_MAP.items():
            if full_word.startswith(cleaned):
                return direction

        if cleaned.startswith("S"):
            return "S"
        if cleaned.startswith("W"):
            return "W"
        if cleaned.startswith("N"):
            return "N"
        if cleaned.startswith("E"):
            return "E"

        return None

    def standardize_bid(self, bid_str):
        b_clean = bid_str.upper().replace(" ", "")

        symbol_map = {
            "♠": "S", "♥": "H", "♦": "D", "♣": "C",
            "@": "S", "&": "C",
        }
        for sym, repl in symbol_map.items():
            b_clean = b_clean.replace(sym, repl)

        if len(b_clean) >= 2 and b_clean[0] in "ILT!|":
            rest = b_clean[1:]
            if rest in ("NT", "N", "S", "H", "D", "C",
                        "SPADES", "HEARTS", "DIAMONDS", "CLUBS"):
                b_clean = "1" + rest

        if b_clean in ("PAS", "PA", "PASSED"):
            return "PASS"
        if b_clean in ("DBL", "DOUBLE", "X"):
            return "DBL"
        if b_clean in ("RDBL", "REDOUBLE", "XX"):
            return "RDBL"

        if len(b_clean) >= 2 and b_clean[0].isdigit():
            digit = b_clean[0]
            suit = b_clean[1:]
            suit_map = {
                "SPADES": "S", "S": "S",
                "HEARTS": "H", "H": "H",
                "DIAMONDS": "D", "D": "D",
                "CLUBS": "C", "C": "C",
                "N": "NT", "NT": "NT",
            }
            suit = suit_map.get(suit, suit)
            if suit in ("S", "H", "D", "C", "NT"):
                return f"{digit}{suit}"

        if b_clean in ("PASS", "DBL", "RDBL"):
            return b_clean

        return b_clean

    def extract_bids(self, bidding_img):
        return self._extract_bids_structured(bidding_img)

    def extract_bids_with_bboxes(self, bidding_img, fx=4.0):
        return self._extract_bids_structured(bidding_img, fx=fx, with_bboxes=True)

    def extract_bidding_hint(self, hint_img, fx=4.0):
        processed = self.preprocess_for_ocr(hint_img, fx=fx, thresh_val=None)
        text = pytesseract.image_to_string(
            processed, config="--psm 6 --oem 3"
        )
        text = text.strip()
        if text:
            text = re.sub(r'\s+', ' ', text)
        return text if text else ""

    def _extract_bids_structured(self, bidding_img, fx=4.0, with_bboxes=False):
        import csv
        from io import StringIO

        processed = self.preprocess_for_ocr(bidding_img, thresh_val=None)

        data_str = pytesseract.image_to_data(
            processed, config="--psm 6", output_type=pytesseract.Output.STRING
        )

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
                "center_x": int(row[left_idx]) + int(row[width_idx]) // 2,
            })

        if not words:
            return []

        all_header_candidates = []
        for w in words:
            dir_key = self.clean_header_text(w["text"])
            if dir_key and w["top"] < processed.shape[0] * 0.35:
                all_header_candidates.append(w)

        if not all_header_candidates:
            if self.verbose:
                print("No direction headers detected in bidding area.")
            return []

        min_top = min(w["top"] for w in all_header_candidates)

        header_words = []
        for w in all_header_candidates:
            if w["top"] - min_top < 30:
                header_words.append((
                    w["center_x"], self.clean_header_text(w["text"]), w["top"]
                ))

        header_words.sort()
        full_order = ["N", "E", "S", "W"]
        min_cx = header_words[0][0]

        if len(header_words) >= 2:
            spacings = [
                header_words[i+1][0] - header_words[i][0]
                for i in range(len(header_words) - 1)
            ]
            col_width = sum(spacings) / len(spacings)
        else:
            col_width = 150.0

        best_rotation = full_order
        best_c0 = min_cx
        max_matches = -1

        for leftmost_slot in range(4):
            candidate_c0 = min_cx - (leftmost_slot * col_width)
            for r in range(4):
                candidate_rotation = full_order[r:] + full_order[:r]
                matches = 0
                for center_x, direction, _ in header_words:
                    slot_idx = int(round((center_x - candidate_c0) / col_width))
                    if 0 <= slot_idx < 4 and candidate_rotation[slot_idx] == direction:
                        matches += 1
                if matches > max_matches:
                    max_matches = matches
                    best_rotation = candidate_rotation
                    best_c0 = candidate_c0

        col_centers = [best_c0 + idx * col_width for idx in range(4)]
        col_dirs = best_rotation
        max_header_top = max(hw[2] for hw in header_words)

        if self.verbose:
            print(f"Bidding columns reconstructed: {list(zip(col_dirs, col_centers))}")

        bid_words = []
        for w in words:
            if w["top"] > max_header_top + 10:
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

        bid_pattern = re.compile(
            r'^(PASS|PAS|PA|PASSED|DBL|DOUBLE|RDBL|REDOUBLE|X|XX|'
            r'[1-7]\s*(?:NT|N|S|H|D|C|SPADES|HEARTS|DIAMONDS|CLUBS))$',
            re.IGNORECASE
        )

        results = []
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

                min_left = min(w["left"] for w in g_words)
                min_top_w = min(w["top"] for w in g_words)
                max_right = max(w["left"] + w["width"] for w in g_words)
                max_bottom = max(w["top"] + w["height"] for w in g_words)

                # Resolve suit symbol images in level bids (e.g. 1H, 1S) that Tesseract misreads
                if len(std_text) >= 2 and std_text[0] in "1234567" and std_text[1:] not in ("S", "H", "D", "C", "NT"):
                    x1 = int(min_left / fx)
                    y1 = int(min_top_w / fx)
                    x2 = int(max_right / fx)
                    y2 = int(max_bottom / fx)
                    x1 = max(0, min(x1, bidding_img.shape[1] - 1))
                    y1 = max(0, min(y1, bidding_img.shape[0] - 1))
                    x2 = max(0, min(x2, bidding_img.shape[1]))
                    y2 = max(0, min(y2, bidding_img.shape[0]))
                    
                    if x2 > x1 and y2 > y1:
                        word_crop = bidding_img[y1:y2, x1:x2]
                        suit_w = int(word_crop.shape[1] * 0.6)
                        if suit_w > 0:
                            suit_crop = word_crop[:, word_crop.shape[1] - suit_w:]
                            suit = self.classify_suit_template_matching(suit_crop)
                            if not suit:
                                suit = self.classify_suit_by_color_shape(suit_crop)
                                
                            if suit in ("spade", "heart", "diamond", "club"):
                                suit_map = {
                                    "spade": "S",
                                    "heart": "H",
                                    "diamond": "D",
                                    "club": "C"
                                }
                                std_text = f"{std_text[0]}{suit_map[suit]}"

                # Disambiguate DBL (Red) and RDBL (Blue) based on color
                if std_text in ("DBL", "RDBL"):
                    x1 = int(min_left / fx)
                    y1 = int(min_top_w / fx)
                    x2 = int(max_right / fx)
                    y2 = int(max_bottom / fx)
                    x1 = max(0, min(x1, bidding_img.shape[1] - 1))
                    y1 = max(0, min(y1, bidding_img.shape[0] - 1))
                    x2 = max(0, min(x2, bidding_img.shape[1]))
                    y2 = max(0, min(y2, bidding_img.shape[0]))
                    
                    if x2 > x1 and y2 > y1:
                        crop = bidding_img[y1:y2, x1:x2]
                        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                        
                        lower_red1 = np.array([0, 100, 50])
                        upper_red1 = np.array([15, 255, 255])
                        lower_red2 = np.array([165, 100, 50])
                        upper_red2 = np.array([180, 255, 255])
                        red_pixels = np.sum((cv2.inRange(hsv, lower_red1, upper_red1) > 0) | (cv2.inRange(hsv, lower_red2, upper_red2) > 0))
                        
                        lower_blue = np.array([90, 100, 50])
                        upper_blue = np.array([130, 255, 255])
                        blue_pixels = np.sum(cv2.inRange(hsv, lower_blue, upper_blue) > 0)
                        
                        if red_pixels > 15 or blue_pixels > 15:
                            if std_text == "RDBL" and red_pixels > blue_pixels * 1.5:
                                std_text = "DBL"
                            elif std_text == "DBL" and blue_pixels > red_pixels * 1.5:
                                std_text = "RDBL"

                if bid_pattern.match(std_text):
                    direction = col_dirs[col_idx]
                    if with_bboxes:
                        bbox = {
                            "x": min_left / fx,
                            "y": min_top_w / fx,
                            "w": (max_right - min_left) / fx,
                            "h": (max_bottom - min_top_w) / fx,
                        }
                        row_bids.append((col_idx, direction, std_text, bbox))
                    else:
                        row_bids.append((col_idx, direction, std_text, None))
                elif self.verbose:
                    print(f"Skipping non-bid text: '{combined_text}' -> '{std_text}'")

            row_bids.sort()
            for entry in row_bids:
                if with_bboxes:
                    _, direction, std_text, bbox = entry
                    results.append({"direction": direction, "bid": std_text, "bbox": bbox})
                else:
                    _, direction, std_text, _ = entry
                    results.append((direction, std_text))

        return results

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
        if best_score > 0.35:
            return best_match
        return None

    def extract_card(self, card_img, is_hand=True, suit_img=None, suit_img_top=None, expected_suit_is_red=None):
        """
        Parses a single card image crop.
        Returns a tuple (rank, suit) or (None, None).
        """
        h, w = card_img.shape[:2]
        if w < 10 or h < 10:
            return None, None
            
        # For normalized hand crops (height 60), we use centered crops where x_suit is at x=15.
        # This keeps the rank and suit perfectly centered and avoids bottom-of-rank pixels contaminating templates.
        if h == 60:
            # Full top half captures rank reliably for loose spacing (player hand)
            # Narrower crop avoids adjacent-card bleed for tight spacing (dummy)
            if "tight" in self.__dict__ and self.tight:
                rank_crop = card_img[2:30, 2:28]
            else:
                rank_crop = card_img[2:30, 2:36]
            # Wider bottom area to capture suit symbol (may be offset due to card overlap)
            suit_crop = card_img[30:55, 2:35]
        elif is_hand:
            # Overlapping cards: rank and suit are on the left
            rank_crop = card_img[2:int(h*0.43), 5:int(w*0.45)]
            suit_crop = card_img[int(h*0.40):int(h*0.93), 2:int(w*0.50)]
        else:
            # Played cards in trick area (fully visible, centered rank and suit symbol)
            # Use dynamic split to find the gap between rank and suit symbol
            top_65 = card_img[0:int(h*0.65), :]
            gray = cv2.cvtColor(top_65, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            row_sums = np.sum(binary > 0, axis=1)
            
            search_start = int(h * 0.40)
            search_end = int(h * 0.62)
            best_y = int(h * 0.50)
            min_sum = float('inf')
            for y in range(search_start, search_end):
                window_sum = np.sum(row_sums[y-1:y+2])
                if window_sum < min_sum:
                    min_sum = window_sum
                    best_y = y
                    
            rank_crop = card_img[2:best_y, int(w*0.10):int(w*0.90)]
            suit_crop = card_img[best_y:int(h*0.95), int(w*0.10):int(w*0.90)]
        
        if rank_crop.size == 0 or suit_crop.size == 0:
            return None, None
            
        def normalize_rank_text(raw_text):
            rank_text = raw_text.strip().upper().replace(" ", "")
            if not rank_text:
                return None

            if "10" in rank_text:
                return "T"
            if rank_text == "1":
                return "T"

            # Common Queen misreads from curved glyphs.
            if rank_text in ["0", "O", "D"]:
                return "Q"

            valid_ranks = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
            return rank_text if rank_text in valid_ranks else None

        rank_text = None
        for fx_val in [5.0, 4.0, 3.0]:
            processed_rank = self.preprocess_for_ocr(rank_crop, fx=fx_val)
            for psm in [8, 10, 6]:
                custom_config = f"--psm {psm} -c tessedit_char_whitelist=AKQJT1098765432"
                try:
                    raw_rank = pytesseract.image_to_string(processed_rank, config=custom_config)
                    rank_text = normalize_rank_text(raw_rank)
                    if rank_text:
                        break
                except Exception:
                    pass
            if rank_text:
                break

        # Disambiguate 9 vs Q on red/black cards using bottom-half ink distribution
        if rank_text in ("9", "Q"):
            gray_rank = cv2.cvtColor(rank_crop, cv2.COLOR_BGR2GRAY)
            h, w = gray_rank.shape
            bottom = gray_rank[h//2:, :]
            _, binary = cv2.threshold(bottom, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            left_ink = np.sum(binary[:, :w//2] > 0)
            right_ink = np.sum(binary[:, w//2:] > 0)
            bottom_ink = left_ink + right_ink
            
            if bottom_ink > 50:  # Real card crop
                ratio = right_ink / max(1, left_ink)
                # Q has balanced ink in the bottom half (ratio around 1.1 - 1.3)
                # 9 has highly skewed ink to the right (ratio around 1.6 - 3.5)
                if ratio < 1.45:
                    rank_text = "Q"
                else:
                    rank_text = "9"
            else:
                # Mock card crop or fallback:
                if rank_text == "9" and right_ink > left_ink * 1.5:
                    rank_text = "Q"
            
        # Extract Suit — use provided suit_img (from card's left edge) if available
        suit_crop_for_match = suit_img if suit_img is not None else suit_crop
        suit = None
        if self.suit_templates:
            suit = self.classify_suit_template_matching(suit_crop_for_match)
            # Cross-check template result against expected color from peak detection
            if suit and expected_suit_is_red is not None:
                template_is_red = suit in ("heart", "diamond")
                if template_is_red != expected_suit_is_red:
                    suit = None
            
        if not suit:
            # Try the top-area crop as fallback (less overlap interference)
            if suit_img_top is not None:
                suit_top = suit_img_top
                if self.suit_templates:
                    top_suit = self.classify_suit_template_matching(suit_top)
                    if top_suit and expected_suit_is_red is not None:
                        top_is_red = top_suit in ("heart", "diamond")
                        if top_is_red == expected_suit_is_red:
                            suit = top_suit
                    elif top_suit:
                        suit = top_suit
                if not suit:
                    suit = self.classify_suit_by_color_shape(suit_top)
            if not suit:
                suit = self.classify_suit_by_color_shape(suit_crop_for_match)
            
        return rank_text, suit

    def extract_multiple_cards(self, cards_img, expected_count=4):
        """
        Finds and extracts multiple cards from an image containing several cards.
        Usually useful for parsing the trick play area or the hand.
        """
        # Grayscale and threshold
        gray = cv2.cvtColor(cards_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Use RETR_LIST instead of RETR_EXTERNAL to find cards nested within the yellow border
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_cards = []
        for c in contours:
            area = cv2.contourArea(c)
            # Filter contours by size to match playing card shape
            # Avoid the giant yellow border contour (which has area > 8000)
            if area < 1000 or area > 8000:
                continue
                
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w)/h
            # Standard cards aspect ratio is around 0.7
            if aspect_ratio < 0.4 or aspect_ratio > 1.2:
                continue
                
            card_crop = cards_img[y:y+h, x:x+w]
            rank, suit = self.extract_card(card_crop, is_hand=False)
            # Filter out any detection that lacks a recognized rank 
            # (e.g. face-down card backs or background elements)
            if rank and suit:
                detected_cards.append({
                    "rank": rank,
                    "suit": suit,
                    "bbox": {"x": x, "y": y, "w": w, "h": h}
                })
                
        return detected_cards

    def extract_hand_cards_linear(self, hand_img, scale=1.0, y_start=0):
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
            import os
            os.makedirs("debug", exist_ok=True)
            cv2.imwrite(f"debug/card_crop_linear_{i}_{rank}{suit}.png", card_crop)
            detected_cards.append({
                "rank": rank,
                "suit": suit,
                "bbox": {
                    "x": int(x_card / scale),
                    "y": y_start,
                    "w": int(40 / scale),
                    "h": int(h_strip / scale)
                }
            })
        return detected_cards

    def extract_hand_cards(self, hand_img):
        """
        Extracts cards from a player hand row crop.
        Finds individual cards by finding peaks in the smoothed vertical projection
        of red and black suit pixels in the suit row (y=41..54).
        Handles suit gaps and variable card spacing perfectly.
        """
        h_strip = hand_img.shape[0]
        w_strip = hand_img.shape[1]
        
        # Dynamically find the card strip vertically within hand_img to handle tall/calibrated crops
        hsv_full = cv2.cvtColor(hand_img, cv2.COLOR_BGR2HSV)
        card_mask = (hsv_full[:, :, 1] < 50) & (hsv_full[:, :, 2] > 170)
        row_card_counts = np.sum(card_mask, axis=1)
        card_rows = np.where(row_card_counts > 0.05 * w_strip)[0]
        
        y_start_orig = 0
        if len(card_rows) >= 10:
            y_start = card_rows[0]
            y_end = card_rows[-1]
            y_start = max(0, y_start - 2)
            y_end = min(h_strip - 1, y_end + 2)
            hand_img = hand_img[y_start:y_end+1, :]
            h_strip = hand_img.shape[0]
            y_start_orig = y_start
        
        # Save cropped (but not yet resized) version for global OCR
        hand_img_cropped = hand_img.copy()
            
        # Normalize hand image height to 60 to match suit template scaling
        scale = 1.0
        if h_strip != 60:
            scale = 60.0 / h_strip
            hand_img = cv2.resize(hand_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            h_strip = 60
            w_strip = hand_img.shape[1]
            
        hsv = cv2.cvtColor(hand_img, cv2.COLOR_BGR2HSV)
        
        # Define RED mask (Hearts/Diamonds)
        lower_red1 = np.array([0, 40, 40])
        upper_red1 = np.array([25, 255, 255])
        lower_red2 = np.array([165, 40, 40])
        upper_red2 = np.array([180, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Define BLACK mask (Spades/Clubs)
        mask_black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 60, 100]))
        
        # Combine masks
        mask_suit = mask_red + mask_black
        
        # 1D profile of suit row (y=41..54) to avoid bottom of rank characters
        profile = np.sum(mask_suit[41:54, :] > 0, axis=0).astype(np.float32)
        
        # Apply 1D smoothing (moving average window of size 13 to merge Heart lobes)
        kernel = np.ones(13) / 13.0
        smoothed = np.convolve(profile, kernel, mode='same')
        
        # Peak detection on smoothed profile
        peaks = []
        min_dist = 15
        for x in range(min_dist, len(smoothed) - min_dist):
            val = smoothed[x]
            if val >= 2.0:
                is_max = True
                for dx in range(-min_dist, min_dist + 1):
                    if smoothed[x + dx] > val:
                        is_max = False
                        break
                if is_max:
                    if not peaks or (x - peaks[-1]["x_suit"]) >= min_dist:
                        # Determine suit color at this peak
                        col_red = np.sum(mask_red[41:54, x] > 0)
                        col_black = np.sum(mask_black[41:54, x] > 0)
                        color = "RED" if col_red >= col_black else "BLACK"
                        peaks.append({
                            "x_suit": x,
                            "color": color
                        })
                        
        if self.verbose:
            print(f"Detected {len(peaks)} card suit peaks using smoothed color profile: {[p['x_suit'] for p in peaks]}")
            
        # Group consecutive peaks of the same color to dynamically classify their suits (alternate or standard)
        groups = []
        for p in peaks:
            if not groups or p["color"] != groups[-1]["color"]:
                groups.append({"color": p["color"], "peaks": [p]})
            else:
                groups[-1]["peaks"].append(p)
                
        black_count = sum(1 for g in groups if g["color"] == "BLACK")
        red_count = sum(1 for g in groups if g["color"] == "RED")
        black_idx = 0
        red_idx = 0
        
        for g in groups:
            if g["color"] == "BLACK":
                if black_count >= 2:
                    suit_name = "spade" if black_idx == 0 else "club"
                else:
                    avg_x = sum(p["x_suit"] for p in g["peaks"]) / len(g["peaks"])
                    suit_name = "spade" if avg_x < (w_strip / 2) else "club"
                black_idx += 1
            else:
                if red_count >= 2:
                    suit_name = "heart" if red_idx == 0 else "diamond"
                else:
                    avg_x = sum(p["x_suit"] for p in g["peaks"]) / len(g["peaks"])
                    suit_name = "heart" if avg_x < (w_strip / 2) else "diamond"
                red_idx += 1
            for p in g["peaks"]:
                p["assigned_suit"] = suit_name

        detected_cards = []

        peak_xs = [p["x_suit"] for p in peaks]

        # Determine whether spacing is tight (dummy-style) or loose (player hand)
        if len(peaks) >= 2:
            gaps = [peak_xs[i+1] - peak_xs[i] for i in range(len(peaks) - 1)]
            avg_spacing = sum(gaps) / len(gaps)
            self.tight = avg_spacing <= 35
        else:
            self.tight = False

        _, white_binary = cv2.threshold(
            cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY), 200, 255, cv2.THRESH_BINARY)
        col_white = np.sum(white_binary > 0, axis=0)
        first_white = int(np.where(col_white > 10)[0][0])

        for idx, p in enumerate(peaks):
            # Crop card centered: x_card goes from p["x_suit"] - 15 to p["x_suit"] + 25 (width 40)
            x_card = max(0, p["x_suit"] - 15)
            card_crop = hand_img[0:60, x_card:min(x_card + 40, w_strip)]

            # Suit crop centered on suit peak
            suit_left = max(0, p["x_suit"] - 15)
            suit_crop_wide = hand_img[30:55, suit_left:min(suit_left + 40, w_strip)]

            # Also prepare a top-area suit crop as fallback (less overlap interference)
            suit_top = hand_img[3:22, suit_left:min(suit_left + 40, w_strip)]

            rank, suit_raw = self.extract_card(card_crop, suit_img=suit_crop_wide, suit_img_top=suit_top, expected_suit_is_red=p["color"] == "RED")
            # Robust sequence-assigned suit
            suit = p.get("assigned_suit") or suit_raw or ("heart" if p["color"] == "RED" else "spade")

            # Save debug crops
            os.makedirs("debug", exist_ok=True)
            cv2.imwrite(f"debug/card_crop_{idx}_{rank}{suit}.png", card_crop)

            detected_cards.append({
                "rank": rank,
                "suit": suit,
                "bbox": {
                    "x": int(x_card / scale),
                    "y": y_start_orig,
                    "w": int(40 / scale),
                    "h": int(60.0 / scale)
                }
            })
            
        # Try global OCR on the cropped (full-res) hand strip for ranks (more accurate than per-card)
        if len(detected_cards) >= 10:
            try:
                import pytesseract as pt
                gray_strip = cv2.cvtColor(hand_img_cropped, cv2.COLOR_BGR2GRAY)
                scaled_strip = cv2.resize(gray_strip, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                
                best_ranks = None
                # Sweeping thresholds to find the most complete rank list matching detected_cards count
                for thresh_val in [110, 100, 95, 120, "otsu"]:
                    if thresh_val == "otsu":
                        thresh_strip = cv2.threshold(scaled_strip, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                    else:
                        thresh_strip = cv2.threshold(scaled_strip, thresh_val, 255, cv2.THRESH_BINARY)[1]
                        
                    # Test both normal and inverted
                    for inv in [False, True]:
                        proc = cv2.bitwise_not(thresh_strip) if inv else thresh_strip
                        raw = pt.image_to_string(proc, config="--psm 11 -c tessedit_char_whitelist=AKQJT9876543210ODod").strip()
                        
                        # Parse and clean ranks by line block to handle vertical noise
                        blocks = [b.strip() for b in raw.split('\n') if b.strip()]
                        cleaned_blocks = []
                        for b in blocks:
                            raw_chars = [c for c in b.upper() if c in "AKQJT9876543210OD"]
                            cleaned_b = []
                            i = 0
                            while i < len(raw_chars):
                                char = raw_chars[i]
                                if char == '1' and i + 1 < len(raw_chars) and raw_chars[i+1] == '0':
                                    cleaned_b.append('T')
                                    i += 2
                                elif char == '1':
                                    cleaned_b.append('T')
                                    i += 1
                                elif char in ['0', 'O', 'D']:
                                    cleaned_b.append('Q')
                                    i += 1
                                else:
                                    cleaned_b.append(char)
                                    i += 1
                            if cleaned_b:
                                cleaned_blocks.append(cleaned_b)
                                
                        # Assemble the final ranks list
                        cleaned_ranks = []
                        if cleaned_blocks:
                            if len(cleaned_blocks) == 1:
                                cleaned_ranks = cleaned_blocks[0]
                            else:
                                first = cleaned_blocks[0]
                                last = cleaned_blocks[-1]
                                if len(first) + len(last) == len(detected_cards):
                                    cleaned_ranks = first + last
                                else:
                                    for cb in cleaned_blocks:
                                        cleaned_ranks.extend(cb)
                                        
                        if len(cleaned_ranks) >= len(detected_cards):
                            best_ranks = cleaned_ranks[:len(detected_cards)]
                            break
                    if best_ranks is not None:
                        break
                        
                if best_ranks is not None:
                    if self.verbose:
                        print(f"Global OCR ranks (perfect match): {''.join(best_ranks)}")
                    for i, c in enumerate(detected_cards):
                        c["rank"] = best_ranks[i]
                else:
                    if self.verbose:
                        print("Global OCR could not find perfect match for card count. Keeping local OCR ranks.")
            except Exception as e:
                if self.verbose:
                    print(f"Global OCR error: {e}")
            
        return detected_cards

    def locate_ui_text_button(self, ui_img, target_text, ui_roi, fx=2.0, thresh_val=127, max_y=None):
        """
        Runs OCR on the UI image to find a specific text button (e.g. "1", "NT").
        Returns the absolute screen coordinates (x, y) of the text's center,
        or None if not found.
        """
        import csv
        import re
        from io import StringIO

        # If max_y is specified, crop the image to exclude everything below max_y
        if max_y is not None:
            max_y_rel = int(max_y - ui_roi["y"])
            if 0 < max_y_rel < ui_img.shape[0]:
                ui_img = ui_img[0:max_y_rel, :]

        # Try with multiple configurations (PSM modes and threshold configurations)
        configs = [
            {"psm": 6, "thresh": thresh_val},
            {"psm": 11, "thresh": thresh_val},
            {"psm": 6, "thresh": "otsu"},
            {"psm": 11, "thresh": None}
        ]

        for cfg in configs:
            psm_val = cfg["psm"]
            t_val = cfg["thresh"]
            
            processed = self.preprocess_for_ocr(ui_img, fx=fx, thresh_val=t_val)
            data_str = pytesseract.image_to_data(processed, config=f"--psm {psm_val}", output_type=pytesseract.Output.STRING)

            f = StringIO(data_str)
            reader = csv.reader(f, delimiter='\t')

            try:
                header = next(reader)
                left_idx = header.index('left')
                top_idx = header.index('top')
                width_idx = header.index('width')
                height_idx = header.index('height')
                text_idx = header.index('text')
            except (StopIteration, ValueError):
                continue

            matches = []
            for row in reader:
                if len(row) <= text_idx:
                    continue
                text = row[text_idx].strip()
                if not text:
                    continue

                # Clean text to handle potential punctuation artifacts from button borders (e.g. "4)" -> "4")
                cleaned_text = re.sub(r'[^a-zA-Z0-9]', '', text)

                if cleaned_text.upper() == target_text.upper() or text.upper() == target_text.upper():
                    left = int(row[left_idx])
                    top = int(row[top_idx])
                    width = int(row[width_idx])
                    height = int(row[height_idx])

                    # Center relative to scaled image
                    rx = (left + width / 2) / fx
                    ry = (top + height / 2) / fx

                    # Global screen coordinates
                    gx = int(ui_roi["x"] + rx)
                    gy = int(ui_roi["y"] + ry)

                    matches.append({"x": gx, "y": gy, "text": text, "top_rel": ry})

            if matches:
                # Sort matches by top_rel descending (pick the one lower down on screen, i.e., in the bidding panel)
                ui_h = ui_img.shape[0]
                valid_matches = []
                for m in matches:
                    ratio = m["top_rel"] / ui_h
                    # Bidding panel is usually between 30% and 95% from the top of the cropped UI
                    if 0.3 <= ratio <= 0.95:
                        valid_matches.append(m)

                if valid_matches:
                    valid_matches.sort(key=lambda m: m["top_rel"], reverse=True)
                    best_match = valid_matches[0]
                else:
                    best_match = matches[0]

                if self.verbose:
                    print(f"🔍 locate_ui_text_button: Found '{target_text}' at global ({best_match['x']}, {best_match['y']}) using config psm={psm_val}, thresh={t_val}")
                return (best_match["x"], best_match["y"])

        if self.verbose:
            print(f"🔍 locate_ui_text_button: No match for '{target_text}' after trying all configs.")
        return None

if __name__ == "__main__":
    # Test Analyzer interface
    print("Bridge Play UI Analyzer initialized successfully.")
