#!/usr/bin/env python3
"""
Read captured images using raw Tesseract OCR.
"""

import os
import cv2
import pytesseract

def preprocess_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return thresh

def main():
    print("====================================================")
    print("                TESSERACT OCR READER                ")
    print("====================================================")

    # 1. Bidding History
    bidding_path = "debug_captures/2_bidding.png"
    if os.path.exists(bidding_path):
        img = cv2.imread(bidding_path)
        processed = preprocess_img(img)
        
        # Run OCR with standard page segmentation mode 6 (single uniform block of text)
        text = pytesseract.image_to_string(processed, config="--psm 6")
        print("\n📝 [1. Bidding Area Raw OCR text]")
        print("----------------------------------------------------")
        print(text.strip())
        print("----------------------------------------------------")
    else:
        print(f"❌ {bidding_path} not found.")

    # 2. Player's Hand Cards
    hand_path = "debug_captures/4_player_hand.png"
    if os.path.exists(hand_path):
        img = cv2.imread(hand_path)
        
        # In player hand, let's run OCR on the entire strip first
        processed = preprocess_img(img)
        text = pytesseract.image_to_string(processed, config="--psm 6")
        print("\n🃏 [2. Player Hand Raw OCR text (Full Strip)]")
        print("----------------------------------------------------")
        print(text.strip())
        print("----------------------------------------------------")
        
        # Now run OCR on each sliced card
        # Let's slice the 13 cards using our dynamic edge layout
        # (X = 0..339, width 339, 13 cards, step = 24.9)
        x_start = 0
        step = 24.9
        h = img.shape[0]
        
        print("\n🗂️ [3. Player Hand Sliced Card Ranks OCR]")
        print("----------------------------------------------------")
        ranks = []
        for i in range(13):
            x_card = int(x_start + i * step)
            # Crop card rank (y from 0 to 43%, x from 5 to 45% of card width 40)
            card_crop = img[0:h, x_card:min(x_card + 40, img.shape[1])]
            rank_crop = card_crop[2:int(h*0.43), 5:int(h*0.45)] # using h as w is fine
            
            # Preprocess and OCR
            gray_rank = cv2.cvtColor(rank_crop, cv2.COLOR_BGR2GRAY)
            scaled_rank = cv2.resize(gray_rank, (0, 0), fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
            thresh_rank = cv2.threshold(scaled_rank, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            
            # Whitelist standard ranks
            config = "--psm 10 -c tessedit_char_whitelist=AKQJT1098765432"
            char = pytesseract.image_to_string(thresh_rank, config=config).strip().upper()
            ranks.append(char if char else "?")
            
        print(" -> ".join(ranks))
        print("----------------------------------------------------")
    else:
        print(f"❌ {hand_path} not found.")

if __name__ == "__main__":
    main()
