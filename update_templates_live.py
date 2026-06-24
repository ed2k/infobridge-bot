#!/usr/bin/env python3
"""
Capture live screen and update rank templates from real game state.
Extracts card crops from live player hand and uses PaddleOCR to verify ranks.
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from capture import ScreenCapture
from analyzer import BridgeAnalyzer

try:
    from paddle_ocr import ocr_text as paddle_ocr_text
    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False


def get_rank_ocr(rank_crop):
    """Get rank from a crop using PaddleOCR."""
    if not HAS_PADDLE:
        return None
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
    except Exception:
        pass
    return None


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
    templates_dir = "templates"
    os.makedirs(templates_dir, exist_ok=True)

    print("Capturing live screen...")
    cap = ScreenCapture()
    analyzer = BridgeAnalyzer(verbose=False)

    hand_img = cap.capture_player_hand()
    if hand_img is None:
        print("Failed to capture player hand.")
        return

    os.makedirs("debug", exist_ok=True)
    cv2.imwrite("debug/live_hand_capture.png", hand_img)
    print(f"Saved live hand capture: debug/live_hand_capture.png ({hand_img.shape})")

    detected_cards = analyzer.extract_hand_cards(hand_img)
    if not detected_cards:
        print("No cards detected in live hand.")
        return

    print(f"Detected {len(detected_cards)} cards.")

    candidates = {}

    for i, card in enumerate(detected_cards):
        rank = card.get("rank")
        if not rank:
            continue

        bbox = card.get("bbox", {})
        x, y, w, h = int(bbox.get("x", 0)), int(bbox.get("y", 0)), int(bbox.get("w", 40)), int(bbox.get("h", 60))

        card_crop = hand_img[y:y+h, x:x+w] if y+h <= hand_img.shape[0] and x+w <= hand_img.shape[1] else None
        if card_crop is None or card_crop.size == 0:
            continue

        if h == 60:
            rank_crop = card_crop[2:38, 2:36]
        else:
            rank_crop = card_crop[2:int(h*0.43), 5:int(w*0.45)]

        if rank_crop.size == 0:
            continue

        ocr_rank = get_rank_ocr(rank_crop)

        if ocr_rank and ocr_rank != rank:
            print(f"  Card {i+1}: detected={rank}, OCR={ocr_rank} (using OCR)")
            rank = ocr_rank
        elif ocr_rank == rank:
            print(f"  Card {i+1}: {rank} (verified by OCR)")
        else:
            print(f"  Card {i+1}: {rank} (OCR unavailable, using detected)")

        quality = compute_quality(rank_crop)

        if rank not in candidates or quality > candidates[rank]["quality"]:
            candidates[rank] = {"crop": rank_crop, "quality": quality, "source": f"live_card_{i}"}

    print("\nSaving templates from live capture...")
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

            print(f"  {rank}: quality={info['quality']:.1f}")
            saved += 1
        else:
            print(f"  {rank}: not found in live hand")

    print(f"\nUpdated {saved}/{len(all_ranks)} templates from live capture.")


if __name__ == "__main__":
    main()
