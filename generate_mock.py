#!/usr/bin/env python3
"""
Generate Mock Bridge Board.
Captures live screen regions and composites them into a sample_board.png,
or generates a synthetic mock if live capture fails.
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from capture import ScreenCapture
from PIL import Image, ImageDraw, ImageFont


def get_font(size):
    """Loads a macOS system TrueType font or falls back to default."""
    for path in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Courier New.ttf"
    ]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_card(img, draw, font, x, y, w, h, rank, suit, suit_color):
    draw.rounded_rectangle([x, y, x+w, y+h], radius=6, fill="white", outline="#e0e0e0", width=2)
    y_offset = 9 if h == 60 else 4
    x_offset = 10 if h == 60 else 6
    draw.text((x + x_offset, y + y_offset), rank, font=font, fill=suit_color)

    suit_y_offset = 13 if h == 60 else 10
    cx, cy = x + w//2, y + h//2 + suit_y_offset

    template_path = f"templates/{suit}.png"
    if os.path.exists(template_path):
        from PIL import ImageOps
        s_img = Image.open(template_path).convert("L")
        color_fill = (255, 0, 0) if suit_color == "red" else (0, 0, 0)
        colored_suit = Image.new("RGB", s_img.size, color=color_fill)
        mask = ImageOps.invert(s_img)
        img.paste(colored_suit, (cx - s_img.size[0]//2, cy - s_img.size[1]//2), mask=mask)
    else:
        size = 12
        if suit == "spade":
            draw.polygon([(cx-3, cy+size), (cx+3, cy+size), (cx, cy+size//2)], fill=suit_color)
            draw.ellipse([cx-size//2, cy, cx, cy+size//2], fill=suit_color)
            draw.ellipse([cx, cy, cx+size//2, cy+size//2], fill=suit_color)
            draw.polygon([(cx-size//2, cy+size//4), (cx+size//2, cy+size//4), (cx, cy-size//2)], fill=suit_color)
        elif suit == "heart":
            draw.ellipse([cx-size//2, cy-size//2, cx, cy], fill=suit_color)
            draw.ellipse([cx, cy-size//2, cx+size//2, cy], fill=suit_color)
            draw.polygon([(cx-size//2, cy-1), (cx+size//2, cy-1), (cx, cy+size//2+2)], fill=suit_color)
        elif suit == "diamond":
            draw.polygon([(cx, cy-size), (cx+size, cy), (cx, cy+size), (cx-size, cy)], fill=suit_color)
        elif suit == "club":
            draw.polygon([(cx-3, cy+size), (cx+3, cy+size), (cx, cy+size//2)], fill=suit_color)
            draw.ellipse([cx-size//2-2, cy-size//4, cx-2, cy+size//4], fill=suit_color)
            draw.ellipse([cx+2, cy-size//4, cx+size//2+2, cy+size//4], fill=suit_color)
            draw.ellipse([cx-size//3, cy-size//2-2, cx+size//3, cy-size//4], fill=suit_color)


def capture_live_composite():
    """Capture live screen regions and composite into a single image."""
    try:
        cap = ScreenCapture()

        bidding_img = cap.capture_bidding()
        trick_img = cap.capture_trick()
        hand_img = cap.capture_player_hand()
        hint_img = cap.capture_bidding_hint()

        panel = cap.capture_game_panel()
        panel_pil = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))

        return panel_pil
    except Exception as e:
        print(f"Live capture failed: {e}")
        return None


def generate_synthetic():
    """Generate synthetic mock board."""
    title_font = get_font(24)
    section_font = get_font(18)
    table_font = get_font(16)
    card_font = get_font(22)
    hand_card_font = get_font(18)
    button_font = get_font(16)

    img = Image.new("RGB", (1200, 800), color="#1e5631")
    draw = ImageDraw.Draw(img)

    draw.text((30, 20), "InfoBridge Bot - Mock Test Screen", font=title_font, fill="white")

    bx, by, bw, bh = 800, 80, 350, 250
    draw.rectangle([bx, by, bx+bw, by+bh], fill="#2e3138", outline="#4e525a", width=3)
    draw.text((bx+20, by+15), "BIDDING HISTORY", font=section_font, fill="#00ff00")
    draw.line([bx+10, by+40, bx+bw-10, by+40], fill="#4e525a", width=2)
    draw.text((bx+30, by+55), "WEST    NORTH   EAST    SOUTH", font=table_font, fill="white")
    draw.text((bx+30, by+90), "Pass    1NT     Pass    2C", font=table_font, fill="white")
    draw.text((bx+30, by+120), "Pass    3NT     Pass    Pass", font=table_font, fill="white")
    draw.text((bx+30, by+150), "DBL     Pass    Pass    Pass", font=table_font, fill="white")

    tx, ty, tw, th = 300, 250, 400, 300
    draw.rectangle([tx, ty, tx+tw, ty+th], fill="#163f24", outline="#226037", width=2)
    draw.text((tx+15, ty+15), "TRICK AREA", font=section_font, fill="#90ee90")

    draw_card(img, draw, card_font, 470, 275, 55, 80, "A", "spade", "black")
    draw_card(img, draw, card_font, 470, 445, 55, 80, "10", "club", "black")
    draw_card(img, draw, card_font, 560, 360, 55, 80, "K", "heart", "red")
    draw_card(img, draw, card_font, 380, 360, 55, 80, "Q", "diamond", "red")

    hx, hy, hw, hh = 300, 600, 500, 60
    draw.text((hx+15, hy-25), "YOUR HAND", font=section_font, fill="#90ee90")
    draw.rectangle([hx, hy, hx+hw, hy+hh], fill="#163f24", outline="#226037", width=2)

    hand_cards = [
        ("A", "spade", "black"), ("K", "spade", "black"), ("Q", "heart", "red"),
        ("J", "heart", "red"), ("10", "heart", "red"), ("9", "diamond", "red"),
        ("8", "diamond", "red"), ("7", "diamond", "red"), ("6", "club", "black"),
        ("5", "club", "black"), ("4", "club", "black"), ("3", "spade", "black"),
        ("2", "heart", "red")
    ]
    for i, (rank, suit, color) in enumerate(hand_cards):
        cx = 320 + i * 32
        draw_card(img, draw, hand_card_font, cx, 600, 45, 60, rank, suit, color)

    return img


def main():
    os.makedirs("debug", exist_ok=True)
    output_path = os.path.join("debug", "sample_board.png")

    print("Attempting live screen capture...")
    live_img = capture_live_composite()

    if live_img is not None:
        live_img.save(output_path)
        print(f"Saved live capture: {os.path.abspath(output_path)}")
        print(f"Size: {live_img.size}")
    else:
        print("Live capture failed, generating synthetic mock...")
        synthetic_img = generate_synthetic()
        synthetic_img.save(output_path)
        print(f"Saved synthetic mock: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()


