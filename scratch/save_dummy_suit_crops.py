import cv2
import numpy as np
import os

def main():
    img_path = "debug_captures/1_ui_full.png"
    if not os.path.exists(img_path):
        print("❌ UI image not found.")
        return
        
    img = cv2.imread(img_path)
    crop = img[275:310, 0:510]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    # Define RED and BLACK masks
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([25, 255, 255])
    lower_red2 = np.array([165, 40, 40])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mask_total = (gray < 220)
    mask_black = mask_total & (mask_red == 0)
    
    red_proj = np.sum(mask_red > 0, axis=0)
    black_proj = np.sum(mask_black > 0, axis=0)
    total_proj = red_proj + black_proj
    
    # Solid borders
    is_border = (total_proj >= 30)
    edge_mask = np.zeros_like(is_border)
    edge_mask[:5] = True
    edge_mask[505:] = True
    
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
    
    # We expect spade, heart, club, diamond (or standard order)
    # Let's check suits_order
    col3_x1 = int(220 * (510 / 510.0))
    col3_x2 = int(330 * (510 / 510.0))
    col4_x1 = int(330 * (510 / 510.0))
    col4_x2 = int(480 * (510 / 510.0))
    
    crop_col3 = crop[:, col3_x1:col3_x2]
    crop_col4 = crop[:, col4_x1:col4_x2]
    
    b3, g3, r3 = cv2.split(crop_col3)
    red3 = np.sum((r3.astype(int) > g3.astype(int) + 40) & (r3.astype(int) > b3.astype(int) + 40))
    
    b4, g4, r4 = cv2.split(crop_col4)
    red4 = np.sum((r4.astype(int) > g4.astype(int) + 40) & (r4.astype(int) > b4.astype(int) + 40))
    
    if red3 >= red4:
        suits_order = ["spade", "heart", "diamond", "club"]
    else:
        suits_order = ["spade", "heart", "club", "diamond"]
        
    expected_colors = []
    for suit in suits_order:
        if suit in ["spade", "club"]:
            expected_colors.append("BLACK")
        else:
            expected_colors.append("RED")
            
    # Greedy matching
    matched_suits = {suit: None for suit in suits_order}
    group_idx = 0
    for suit_idx, suit_name in enumerate(suits_order):
        expected_color = expected_colors[suit_idx]
        while group_idx < len(all_groups):
            g_start, g_end, g_color = all_groups[group_idx]
            if g_color == expected_color:
                matched_suits[suit_name] = (g_start, g_end)
                group_idx += 1
                break
            else:
                break
                
    os.makedirs("debug_ocr_test/dummy_suits", exist_ok=True)
    
    for suit_name in suits_order:
        bounds = matched_suits[suit_name]
        if bounds is None:
            print(f"  {suit_name}: VOID")
            continue
            
        x1, x2 = bounds
        x1_pad = max(0, x1 - 3)
        x2_pad = min(510, x2 + 3)
        
        col_crop = crop[:, x1_pad:x2_pad]
        
        # Save crop
        cv2.imwrite(f"debug_ocr_test/dummy_suits/{suit_name}_crop.png", col_crop)
        print(f"Saved debug_ocr_test/dummy_suits/{suit_name}_crop.png (x={x1}..{x2})")

if __name__ == "__main__":
    main()
