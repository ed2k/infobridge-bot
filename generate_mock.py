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

def draw_card(img, draw, font, x, y, w, h, rank, suit, suit_color):
    # Draw card background (white round-rect)
    draw.rounded_rectangle([x, y, x+w, y+h], radius=6, fill="white", outline="#e0e0e0", width=2)
    
    # Draw rank (e.g. A, K, Q, J, 10)
    # BBO standard: height 60 uses y+9, x+10 to align rank with crop offset. Height 80 uses y+4, x+6
    y_offset = 9 if h == 60 else 4
    x_offset = 10 if h == 60 else 6
    draw.text((x + x_offset, y + y_offset), rank, font=font, fill=suit_color)
    
    # Draw suit shape in center
    # BBO standard: height 60 uses y + h//2 + 13 (relative cy=43) to align with crop y=37..50
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
    draw_card(img, draw, card_font, 470, 275, 55, 80, "A", "spade", "black")  # North
    draw_card(img, draw, card_font, 470, 445, 55, 80, "10", "club", "black")  # South
    draw_card(img, draw, card_font, 560, 360, 55, 80, "K", "heart", "red")    # East
    draw_card(img, draw, card_font, 380, 360, 55, 80, "Q", "diamond", "red")  # West
    
    # 4. Draw Player Hand Area (x=300, y=600, w=500, h=60)
    hx, hy, hw, hh = 300, 600, 500, 60
    draw.text((hx+15, hy-25), "YOUR HAND", font=section_font, fill="#90ee90")
    draw.rectangle([hx, hy, hx+hw, hy+hh], fill="#163f24", outline="#226037", width=2)
    
    # Draw 13 starting hand cards
    hand_cards = [
        ("A", "spade", "black"),
        ("K", "spade", "black"),
        ("Q", "heart", "red"),
        ("J", "heart", "red"),
        ("10", "heart", "red"),
        ("9", "diamond", "red"),
        ("8", "diamond", "red"),
        ("7", "diamond", "red"),
        ("6", "club", "black"),
        ("5", "club", "black"),
        ("4", "club", "black"),
        ("3", "spade", "black"),
        ("2", "heart", "red")
    ]
    for i, (rank, suit, color) in enumerate(hand_cards):
        cx = 320 + i * 32
        draw_card(img, draw, hand_card_font, cx, 600, 45, 60, rank, suit, color)
    
    # 4.5 Draw Bidding Panel (x=750, y=350, w=400, h=115)
    # This represents the active bidding buttons (levels 1-7, suits C, D, H, S, NT)
    # placed on the right side of the table.
    draw.rounded_rectangle([750, 350, 1150, 465], radius=8, fill="#2b2d31", outline="#3f4248", width=2)
    
    # Draw Level Buttons (1 to 7)
    levels = ["1", "2", "3", "4", "5", "6", "7"]
    for i, lvl in enumerate(levels):
        cx = 836 + i * 38
        cy = 380
        draw.rounded_rectangle([cx - 15, cy - 15, cx + 15, cy + 15], radius=4, fill="#4e525a", outline="#6e727a", width=1)
        draw.text((cx - 5, cy - 9), lvl, font=button_font, fill="white")
        
    # Draw Suit Buttons (C, D, H, S, NT)
    # Clubs (C), Diamonds (D), Hearts (H), Spades (S), NT
    # Align centers exactly to match offsets relative to NT=1050: S=1000, H=950, D=900, C=850
    suits = [
        ("C", 850, "black"),
        ("D", 900, "red"),
        ("H", 950, "red"),
        ("S", 1000, "black"),
        ("NT", 1050, "white")
    ]
    
    for suit_name, cx, color in suits:
        cy = 430
        w = 40 if suit_name == "NT" else 30
        draw.rounded_rectangle([cx - w//2, cy - 15, cx + w//2, cy + 15], radius=4, fill="#4e525a", outline="#6e727a", width=1)
        
        # Draw labels
        if suit_name == "NT":
            draw.text((cx - 11, cy - 9), "NT", font=button_font, fill=color)
        else:
            draw.text((cx - 6, cy - 9), suit_name, font=button_font, fill=color)
            
    # Draw special buttons "Pass" and "DBL"
    # Pass
    draw.rounded_rectangle([770, 415, 815, 445], radius=4, fill="#2e7d32", outline="#388e3c", width=1)
    draw.text((774, 424), "PASS", font=button_font, fill="white")
    # DBL
    draw.rounded_rectangle([1085, 415, 1130, 445], radius=4, fill="#c62828", outline="#d32f2f", width=1)
    draw.text((1092, 424), "DBL", font=button_font, fill="white")
    
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

