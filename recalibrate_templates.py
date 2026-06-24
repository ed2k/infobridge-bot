#!/usr/bin/env python3
"""
Recalibrate rank templates using PaddleOCR.
Analyzes card crops from debug/ folder and uses PaddleOCR as ground truth
to extract/update rank templates (26x18 grayscale).
"""

import os
import cv2
import numpy as np
import re

try:
    from paddle_ocr import ocr_text as paddle_ocr_text
    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False
    print("Warning: PaddleOCR not available, falling back to Tesseract only")

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False


def get_rank_ocr(rank_crop):
    """Get rank from a crop using PaddleOCR then Tesseract fallback."""
    if HAS_PADDLE:
        try:
            text = paddle_ocr_text(rank_crop, min_confidence=0.3)
            if text:
                cleaned = text.strip().upper().replace(" ", "")
                if cleaned == "10": cleaned = "T"
                elif cleaned == "1": cleaned = "T"
                elif cleaned in ("0", "O", "D"): cleaned = "Q"
                valid = {"A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"}
                if cleaned in valid:
                    return cleaned
        except Exception as e:
            if verbose:
                print(f"  PaddleOCR failed: {e}")

    if HAS_TESSERACT:
        try:
            scaled = cv2.resize(rank_crop, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            _, thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            config = "--psm 10 -c tessedit_char_whitelist=AKQJT98765432"
            text = pytesseract.image_to_string(thresh, config=config).strip().upper()
            if text == "10": text = "T"
            elif text == "1": text = "T"
            elif text in ("0", "O", "D"): text = "Q"
            valid = {"A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"}
            if text in valid:
                return text
        except Exception:
            pass

    return None


def extract_rank_crop(card_img):
    """Extract rank region from a 60-height normalized card crop."""
    h, w = card_img.shape[:2]
    if h == 60:
        crop = card_img[2:38, 2:36]
    elif h > 20:
        crop = card_img[2:int(h*0.43), 5:int(w*0.45)]
    else:
        return None
    if crop is None or crop.size == 0:
        return None
    ch, cw = crop.shape[:2]
    if ch > 23:
        crop[23:, :] = 255
    return crop


def compute_quality(rank_crop):
    """Score a rank crop by contrast and clarity."""
    if rank_crop is None or rank_crop.size == 0:
        return -1
    std = np.std(rank_crop)
    border_penalty = 0
    if np.any(rank_crop[:, 0] < 50) or np.any(rank_crop[:, -1] < 50):
        border_penalty = 50
    if np.any(rank_crop[0, :] < 50) or np.any(rank_crop[-1, :] < 50):
        border_penalty += 50
    return std - border_penalty


def main():
    debug_dir = "debug"
    templates_dir = "templates"

    if not os.path.exists(debug_dir):
        print(f"Error: {debug_dir}/ directory not found.")
        return

    os.makedirs(templates_dir, exist_ok=True)

    pattern = re.compile(r"card_crop_(?:linear_)?\d+_([AKQJT98765432])([a-z]+)\.png")

    candidates = {}
    verbose = False

    print("Scanning debug card crops...")
    for filename in sorted(os.listdir(debug_dir)):
        match = pattern.match(filename)
        if not match:
            continue

        expected_rank = match.group(1)
        filepath = os.path.join(debug_dir, filename)
        img = cv2.imread(filepath)
        if img is None:
            continue

        rank_crop = extract_rank_crop(img)
        if rank_crop is None:
            continue

        ocr_rank = get_rank_ocr(rank_crop)

        if ocr_rank != expected_rank:
            continue

        quality = compute_quality(rank_crop)
        if quality <= 0:
            continue

        if expected_rank not in candidates or quality > candidates[expected_rank]["quality"]:
            candidates[expected_rank] = {
                "crop": rank_crop,
                "quality": quality,
                "source": filename,
            }

    print("\nSaving calibrated templates...")
    all_ranks = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
    saved = 0

    for rank in all_ranks:
        if rank in candidates:
            info = candidates[rank]
            crop = info["crop"]

            if crop.shape != (26, 18):
                crop = cv2.resize(crop, (18, 26), interpolation=cv2.INTER_AREA)

            for prefix in ["rank_", ""]:
                out_path = os.path.join(templates_dir, f"{prefix}{rank}.png")
                cv2.imwrite(out_path, crop)

            print(f"  {rank}: {info['quality']:.1f} quality from {info['source']}")
            saved += 1
        else:
            print(f"  {rank}: no verified crop found")

    print(f"\nCalibrated {saved}/{len(all_ranks)} rank templates.")


if __name__ == "__main__":
    main()
