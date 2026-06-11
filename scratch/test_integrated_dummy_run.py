import cv2
import numpy as np
import pytesseract
from collections import Counter

def detect_dummy_hands_trace(img):
    if img is None:
        print("Image is None")
        return None
        
    h_img, w_img = img.shape[:2]
    print(f"Image shape: {w_img}x{h_img}")
    
    best_suits = []
    
    if h_img >= 310:
        crop_w = w_img
        dummy_text_crop = img[275:310, 0:crop_w]
        gray_dt = cv2.cvtColor(dummy_text_crop, cv2.COLOR_BGR2GRAY)
        
        configs = [
            (3.0, "otsu", True, 6),
            (4.0, "otsu", True, 6),
        ]
        
        for idx, (fx_val, thresh_val, invert_val, psm_val) in enumerate(configs):
            print(f"\n--- Config {idx+1}: fx={fx_val}, thresh={thresh_val}, invert={invert_val}, psm={psm_val} ---")
            scaled_dt = cv2.resize(gray_dt, (0, 0), fx=fx_val, fy=fx_val, interpolation=cv2.INTER_CUBIC)
            if thresh_val == "otsu":
                thresh_dt = cv2.threshold(scaled_dt, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            else:
                thresh_dt = cv2.threshold(scaled_dt, thresh_val, 255, cv2.THRESH_BINARY)[1]
                
            proc_dt = cv2.bitwise_not(thresh_dt) if invert_val else thresh_dt
            
            try:
                txt = pytesseract.image_to_string(proc_dt, config=f"--psm {psm_val}")
                txt_clean = txt.strip().replace("\n", " ")
                print(f"Raw OCR: '{txt}'")
                print(f"Cleaned raw: '{txt_clean}'")
                if not txt_clean:
                    continue
                    
                mapping = {
                    "0": "Q", "O": "Q", "D": "Q",
                    "1": "T", "S": "J",
                    "N": "T", "W": "T",
                    "Z": "",
                    "E": "6",
                    "M": "", "B": "", "I": "", "F": "", "H": "", "X": "",
                }
                cleaned_ranks = []
                text_upper = txt_clean.upper().replace("10", "T")
                for char in text_upper:
                    if char.isdigit():
                        cleaned_ranks.append(char)
                    elif char in mapping:
                        repl = mapping[char]
                        if repl:
                            cleaned_ranks.append(repl)
                    elif char in ["A", "K", "Q", "J", "T"]:
                        cleaned_ranks.append(char)
                print(f"Cleaned ranks: {cleaned_ranks}")
                        
                filtered_ranks = []
                for c in cleaned_ranks:
                    if not filtered_ranks or filtered_ranks[-1] != c:
                        filtered_ranks.append(c)
                print(f"Filtered ranks: {filtered_ranks}")
                        
                rank_order = {r: idx for idx, r in enumerate(["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"])}
                suits = []
                current_suit = []
                for r in filtered_ranks:
                    if r not in rank_order:
                        continue
                    if not current_suit:
                        current_suit.append(r)
                    else:
                        prev_r = current_suit[-1]
                        if rank_order[r] <= rank_order[prev_r]:
                            suits.append(current_suit)
                            current_suit = [r]
                        else:
                            current_suit.append(r)
                if current_suit:
                    suits.append(current_suit)
                print(f"Suits detected ({len(suits)}): {suits}")
                    
                if len(suits) <= 4:
                    total_cards = sum(len(s) for s in suits)
                    print(f"Acceptable configuration: total_cards={total_cards}, len(suits)={len(suits)}")
                    if 0 < total_cards <= 13:
                        if not best_suits or total_cards > sum(len(s) for s in best_suits):
                            best_suits = suits
                            print(f"Updating best_suits to: {best_suits}")
                            if total_cards == 13 and len(suits) == 4:
                                break
                else:
                    print("Skipping config: len(suits) > 4")
            except Exception as e:
                print(f"Exception: {e}")
                
    if best_suits:
        print(f"\nSUCCESS: best_suits found: {best_suits}")
    else:
        print(f"\nFAILURE: fallback to contours needed")

if __name__ == "__main__":
    img = cv2.imread("debug_captures/1_ui_full.png")
    detect_dummy_hands_trace(img)
