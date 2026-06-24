#!/usr/bin/env python3
"""
Read captured images using OCR.
Supports both PaddleOCR (preferred) and Tesseract as fallback.
"""

import os
import cv2
import pytesseract

try:
    from paddle_ocr import ocr_text as paddle_ocr_text, ocr_image as paddle_ocr_image
    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False

def preprocess_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return thresh

def ocr_image(img, method="auto"):
    """
    Run OCR on an image using the specified method.
    
    Args:
        img: OpenCV BGR image
        method: "paddle", "tesseract", or "auto" (tries PaddleOCR first)
    
    Returns:
        OCR text string
    """
    if method == "auto" and HAS_PADDLE:
        try:
            return paddle_ocr_text(img, min_confidence=0.3)
        except Exception as e:
            print(f"PaddleOCR failed: {e}, falling back to Tesseract")
    
    processed = preprocess_img(img)
    return pytesseract.image_to_string(processed, config="--psm 6")

def main():
    print("====================================================")
    print("                OCR READER                          ")
    print("====================================================")
    print(f"PaddleOCR available: {HAS_PADDLE}")

    # 1. Bidding History
    bidding_path = "debug/2_bidding.png"
    if os.path.exists(bidding_path):
        img = cv2.imread(bidding_path)
        text = ocr_image(img)
        print("\n📝 [1. Bidding Area Raw OCR text]")
        print("----------------------------------------------------")
        print(text.strip())
        print("----------------------------------------------------")
    else:
        print(f"❌ {bidding_path} not found.")

    # 2. Player's Hand Cards
    hand_path = "debug/4_player_hand.png"
    if os.path.exists(hand_path):
        img = cv2.imread(hand_path)
        
        text = ocr_image(img)
        print("\n🃏 [2. Player Hand Raw OCR text (Full Strip)]")
        print("----------------------------------------------------")
        print(text.strip())
        print("----------------------------------------------------")
        
        # Now run OCR on each sliced card
        x_start = 0
        step = 24.9
        h = img.shape[0]
        
        print("\n🗂️ [3. Player Hand Sliced Card Ranks OCR]")
        print("----------------------------------------------------")
        ranks = []
        for i in range(13):
            x_card = int(x_start + i * step)
            card_crop = img[0:h, x_card:min(x_card + 40, img.shape[1])]
            rank_crop = card_crop[2:int(h*0.43), 5:int(h*0.45)]
            
            gray_rank = cv2.cvtColor(rank_crop, cv2.COLOR_BGR2GRAY)
            scaled_rank = cv2.resize(gray_rank, (0, 0), fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
            thresh_rank = cv2.threshold(scaled_rank, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            
            config = "--psm 10 -c tessedit_char_whitelist=AKQJ1098765432"
            char = pytesseract.image_to_string(thresh_rank, config=config).strip().upper()
            ranks.append(char if char else "?")
            
        print(" -> ".join(ranks))
        print("----------------------------------------------------")
    else:
        print(f"❌ {hand_path} not found.")

if __name__ == "__main__":
    main()
