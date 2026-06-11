import cv2
import numpy as np
import pytesseract

def clean_rank(text):
    mapping = {
        "0": "Q", "O": "Q", "D": "Q",
        "S": "J", "N": "T", "W": "T",
        "Z": "", "E": "6",
        "M": "", "B": "", "I": "", "F": "", "H": "", "X": "",
    }
    cleaned = []
    text_upper = text.strip().upper().replace("\n", "").replace(" ", "").replace("10", "T")
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
    return [r for r in filtered if r in rank_order]

def test_tight_crops(img_path):
    print(f"\n--- Testing {img_path} with dynamic tight color crops ---")
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to load image")
        return
        
    crop = img[275:310, 0:510]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    # Define RED mask (Hearts/Diamonds)
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([25, 255, 255])
    lower_red2 = np.array([165, 40, 40])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Define BLACK/GREEN mask dynamically as any text pixel (gray < 220) that is not red
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mask_total = (gray < 220)
    mask_black = mask_total & (mask_red == 0)
    
    # Build a profile of active columns for RED and BLACK
    red_proj = np.sum(mask_red > 0, axis=0)
    black_proj = np.sum(mask_black > 0, axis=0)
    total_proj = red_proj + black_proj
    
    # Solid borders have projection value close to crop height (35)
    is_border = (total_proj >= 30)
    
    # We ignore the very edges (x < 5, x > 505) for sanity
    edge_mask = np.zeros_like(is_border)
    edge_mask[:5] = True
    edge_mask[505:] = True
    
    red_active = (red_proj > 1) & (~is_border) & (~edge_mask)
    black_active = (black_proj > 1) & (~is_border) & (~edge_mask)
    
    # Find contiguous groups of active pixels for RED and BLACK
    def get_groups(active_array):
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
                    # Merge groups separated by small gaps (e.g. less than 40 pixels)
                    if groups and (start - groups[-1][1]) < 40:
                        groups[-1] = (groups[-1][0], x - 1)
                    else:
                        groups.append((start, x - 1))
        if in_group:
            if groups and (start - groups[-1][1]) < 40:
                groups[-1] = (groups[-1][0], len(active_array) - 1)
            else:
                groups.append((start, len(active_array) - 1))
        return groups
        
    red_groups = get_groups(red_active)
    black_groups = get_groups(black_active)
    
    print(f"  Detected Black Groups: {black_groups}")
    print(f"  Detected Red Groups:   {red_groups}")
    
    # We expect 4 suit columns:
    # 1. Spades (Black Group 1)
    # 2. Hearts (Red Group 1)
    # 3. Clubs (Black Group 2)
    # 4. Diamonds (Red Group 2)
    suits = []
    
    # Spades (Black Group 1)
    if len(black_groups) >= 1:
        suits.append(("Spades", black_groups[0]))
    else:
        suits.append(("Spades", None))
        
    # Hearts (Red Group 1)
    if len(red_groups) >= 1:
        suits.append(("Hearts", red_groups[0]))
    else:
        suits.append(("Hearts", None))
        
    # Clubs (Black Group 2)
    if len(black_groups) >= 2:
        suits.append(("Clubs", black_groups[1]))
    elif len(black_groups) == 1 and len(red_groups) >= 1:
        # If only 1 black group is found, it might be Spades or Clubs.
        # But we know Clubs is after Hearts, so if black_groups[0] is after red_groups[0], it's Clubs!
        if black_groups[0][0] > red_groups[0][1]:
            # Move it to Clubs
            suits[0] = ("Spades", None)
            suits.append(("Clubs", black_groups[0]))
        else:
            suits.append(("Clubs", None))
    else:
        suits.append(("Clubs", None))
        
    # Diamonds (Red Group 2)
    if len(red_groups) >= 2:
        suits.append(("Diamonds", red_groups[1]))
    elif len(red_groups) == 1 and len(black_groups) >= 1:
        if red_groups[0][0] > black_groups[-1][1]:
            suits[1] = ("Hearts", None)
            suits.append(("Diamonds", red_groups[0]))
        else:
            suits.append(("Diamonds", None))
    else:
        suits.append(("Diamonds", None))
        
    for name, bounds in suits:
        if bounds is None:
            print(f"  {name}: VOID")
            continue
            
        x1, x2 = bounds
        # Add a small padding of 3 pixels to the crop to avoid cropping card edges too closely
        x1_pad = max(0, x1 - 3)
        x2_pad = min(510, x2 + 3)
        
        col_crop = crop[:, x1_pad:x2_pad]
        gray_col = cv2.cvtColor(col_crop, cv2.COLOR_BGR2GRAY)
        
        fx = 4.0
        scaled = cv2.resize(gray_col, (0, 0), fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
        thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        proc = cv2.bitwise_not(thresh)
        
        try:
            txt = pytesseract.image_to_string(proc, config="--psm 6")
            ranks = clean_rank(txt)
            print(f"  {name} (x={x1}..{x2}): raw='{txt.strip()}' -> ranks={ranks}")
        except Exception as e:
            print(f"  {name} Error: {e}")

if __name__ == "__main__":
    test_tight_crops("debug_captures/live_ui_all_sides.png")
    test_tight_crops("debug_captures/1_ui_full.png")
