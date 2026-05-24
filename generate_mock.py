#!/usr/bin/env python3
"""
Generate Mock Bridge Board.
Creates a sample screenshot image 'sample_board.png' to test OCR and CV card classification.
"""

import os
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

def draw_card(draw, font, x, y, w, h, rank, suit, suit_color):
    # Draw card background (white round-rect)
    draw.rounded_rectangle([x, y, x+w, y+h], radius=6, fill="white", outline="black", width=2)
    
    # Draw rank (e.g. A, K, Q, J, 10)
    # Positioning offset based on font size
    draw.text((x + 6, y + 4), rank, font=font, fill=suit_color)
    
    # Draw suit shape in center
    cx, cy = x + w//2, y + h//2 + 10
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

def main():
    # Load fonts
    title_font = get_font(24)
    section_font = get_font(18)
    table_font = get_font(16)
    card_font = get_font(22)
    hand_card_font = get_font(18)
    button_font = get_font(16)

    # Board size: 1200 x 800
    img = Image.new("RGB", (1200, 800), color="#1e5631")  # Dark green felt table
    draw = ImageDraw.Draw(img)
    
    # 1. Title / Header info
    draw.text((30, 20), "InfoBridge Bot - Mock Test Screen", font=title_font, fill="white")
    
    # 2. Draw Bidding Box (Top-Right area: x=800, y=80, w=350, h=250)
    bx, by, bw, bh = 800, 80, 350, 250
    draw.rectangle([bx, by, bx+bw, by+bh], fill="#2e3138", outline="#4e525a", width=3)
    
    # Draw Bidding Text
    draw.text((bx+20, by+15), "BIDDING HISTORY", font=section_font, fill="#00ff00")
    draw.line([bx+10, by+40, bx+bw-10, by+40], fill="#4e525a", width=2)
    
    # Simple grid and some bids
    draw.text((bx+30, by+55), "WEST    NORTH   EAST    SOUTH", font=table_font, fill="white")
    draw.text((bx+30, by+90), "Pass    1NT     Pass    2C", font=table_font, fill="white")
    draw.text((bx+30, by+120), "Pass    3NT     Pass    Pass", font=table_font, fill="white")
    draw.text((bx+30, by+150), "DBL     Pass    Pass    Pass", font=table_font, fill="white")
    
    # 3. Draw Trick Play Area (x=300, y=250, w=400, h=300)
    tx, ty, tw, th = 300, 250, 400, 300
    draw.rectangle([tx, ty, tx+tw, ty+th], fill="#163f24", outline="#226037", width=2)
    draw.text((tx+15, ty+15), "TRICK AREA", font=section_font, fill="#90ee90")
    
    # Draw four cards played (North, East, South, West)
    draw_card(draw, card_font, 470, 275, 55, 80, "A", "spade", "black")  # North
    draw_card(draw, card_font, 470, 445, 55, 80, "10", "club", "black")  # South
    draw_card(draw, card_font, 560, 360, 55, 80, "K", "heart", "red")    # East
    draw_card(draw, card_font, 380, 360, 55, 80, "Q", "diamond", "red")  # West
    
    # 4. Draw Player Hand Area (x=300, y=600, w=500, h=120)
    hx, hy, hw, hh = 300, 600, 500, 120
    draw.rectangle([hx, hy, hx+hw, hy+hh], fill="#163f24", outline="#226037", width=2)
    draw.text((hx+15, hy+10), "YOUR HAND", font=section_font, fill="#90ee90")
    
    # Draw some cards in the hand
    draw_card(draw, hand_card_font, 340, 635, 45, 70, "A", "heart", "red")
    draw_card(draw, hand_card_font, 395, 635, 45, 70, "J", "spade", "black")
    draw_card(draw, hand_card_font, 450, 635, 45, 70, "10", "spade", "black")
    draw_card(draw, hand_card_font, 505, 635, 45, 70, "9", "diamond", "red")
    draw_card(draw, hand_card_font, 560, 635, 45, 70, "5", "club", "black")
    draw_card(draw, hand_card_font, 615, 635, 45, 70, "3", "club", "black")
    draw_card(draw, hand_card_font, 670, 635, 45, 70, "2", "heart", "red")
    
    # 5. Draw Build Info Button (x=900, y=680, w=200, h=50)
    btn_x, btn_y, btn_w, btn_h = 900, 680, 200, 50
    draw.rounded_rectangle([btn_x, btn_y, btn_x+btn_w, btn_y+btn_h], radius=8, fill="#0275d8", outline="#025aa5", width=2)
    draw.text((btn_x+60, btn_y+18), "Build Info", font=button_font, fill="white")
    
    # Save image
    output_path = "sample_board.png"
    img.save(output_path)
    print(f"✅ Generated mock bridge board screenshot: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()

