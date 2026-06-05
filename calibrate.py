#!/usr/bin/env python3
"""
Interactive Screen Calibration Tool for Bridge Bot.
Saves coordinates of various screen regions to config.json.
"""

import os
import json
import time
import pyautogui

CONFIG_FILE = "config.json"

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

def get_point(prompt):
    print(f"\n👉 {prompt}")
    print("   [Hover mouse over the target and press ENTER in this terminal]")
    input()
    x, y = pyautogui.position()
    # On macOS with retina screens, pyautogui coordinates are in points, 
    # which is exactly what pyautogui/mss expects for positioning.
    print(f"   Saved Coordinate: X={x}, Y={y}")
    return {"x": int(x), "y": int(y)}

def get_roi(name):
    print(f"\n--- Calibrating Region: {name} ---")
    top_left = get_point(f"Point to the TOP-LEFT corner of the {name}")
    bottom_right = get_point(f"Point to the BOTTOM-RIGHT corner of the {name}")
    
    # Calculate box
    x = top_left["x"]
    y = top_left["y"]
    width = bottom_right["x"] - x
    height = bottom_right["y"] - y
    
    if width <= 0 or height <= 0:
        print("⚠️ Warning: Bottom-right is not to the right/below top-left. Let's retry this region.")
        return get_roi(name)
        
    print(f"✅ Region Calibrated: X={x}, Y={y}, Width={width}, Height={height}")
    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height
    }

def main():
    clear_screen()
    print("====================================================")
    # Print a premium, beautifully stylized header
    print("           BRIDGE PLAY UI BOT - CALIBRATION          ")
    print("====================================================")
    print("This utility will help calibrate the screen coordinates for the bot.")
    print("Make sure your Bridge game UI is visible on screen.")
    print("If you make a mistake, you can re-run this script anytime.\n")
    
    input("Press ENTER to start calibration...")
    
    config = {}
    
    # 1. Main Play UI ROI
    config["ui_roi"] = get_roi("Main Bridge Play UI (entire table region)")
    
    # 2. Bidding ROI
    config["bidding_roi"] = get_roi("Bidding History / Bid Box")
    
    # 3. Trick Area ROI
    config["trick_roi"] = get_roi("Trick Area (where the 4 played cards are placed)")
    
    # 4. Player Cards ROI
    config["player_hand_roi"] = get_roi("Player Hand (your own cards at the bottom of the table)")
    
    # 5. Build Info Button
    print("\n--- Calibrating Click Action ---")
    config["build_info_btn"] = get_point("Point to the contract (number and suit) above the card area around the center of the board to click")
    
    # Save config
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)
        
    print("\n====================================================")
    print(f"🎉 Calibration completed successfully!")
    print(f"Configuration saved to: {os.path.abspath(CONFIG_FILE)}")
    print("====================================================")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCalibration cancelled.")
