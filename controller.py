#!/usr/bin/env python3
"""
Mouse Controller Module for Bridge Bot.
Handles moving the mouse to coordinates, clicking, and returning cursor to original position.
"""

import os
import json
import time
import pyautogui

CONFIG_FILE = "config.json"

# Enable PyAutoGUI FailSafe (move mouse to corner to abort)
pyautogui.FAILSAFE = True

class BridgeController:
    def __init__(self, config_path=CONFIG_FILE, dry_run=False):
        self.config_path = config_path
        self.dry_run = dry_run
        self.config = self.load_config()

    def load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Configuration file '{self.config_path}' not found. "
                f"Please run 'python calibrate.py' first."
            )
        with open(self.config_path, "r") as f:
            return json.load(f)

    def click_build_info(self, return_to_start=True, move_duration=0.3):
        """
        Moves the mouse smoothly to the 'Build Info' button, clicks it, 
        and optionally returns the cursor to its starting position.
        """
        self.config = self.load_config()  # Reload in case configuration changed
        
        target = self.config.get("build_info_btn")
        if not target:
            raise ValueError("Build Info button coordinates not configured in config.json")

        target_x = target["x"]
        target_y = target["y"]

        if self.dry_run:
            print(f"🤖 [DRY RUN] Would click Build Info button at ({target_x}, {target_y})")
            return

        # Record starting position
        start_x, start_y = pyautogui.position()
        
        print(f"🖱️ Moving mouse from ({start_x}, {start_y}) to Build Info button at ({target_x}, {target_y})...")
        
        # Move smoothly to button
        pyautogui.moveTo(target_x, target_y, duration=move_duration, tween=pyautogui.easeInOutQuad)
        
        # Small sleep before click to make it reliable
        time.sleep(0.1)
        
        # Click
        print("🖱️ Clicking button...")
        pyautogui.click()
        
        # Small wait after click
        time.sleep(0.2)

        if return_to_start:
            print(f"🖱️ Returning mouse to original position: ({start_x}, {start_y})")
            pyautogui.moveTo(start_x, start_y, duration=move_duration, tween=pyautogui.easeInOutQuad)

    def play_card(self, card_bbox, return_to_start=True, move_duration=0.3):
        """
        Calculates the screen coordinate of a card based on its bounding box
        relative to the player hand ROI, moves the mouse, clicks it, and returns.
        """
        self.config = self.load_config()
        
        hand_roi = self.config.get("player_hand_roi")
        if not hand_roi:
            raise ValueError("Player hand ROI coordinates not configured in config.json")
            
        # Calculate target global coordinates
        # Card center is relative x + (w / 2), and y is center of hand strip
        target_x = int(hand_roi["x"] + card_bbox["x"] + (card_bbox["w"] / 2))
        target_y = int(hand_roi["y"] + (hand_roi["height"] / 2))
        
        if self.dry_run:
            print(f"🤖 [DRY RUN] Would click card at ({target_x}, {target_y}) - relative bbox {card_bbox}")
            return
            
        start_x, start_y = pyautogui.position()
        print(f"🖱️ Moving mouse to card: ({target_x}, {target_y}) from ({start_x}, {start_y})...")
        
        pyautogui.moveTo(target_x, target_y, duration=move_duration, tween=pyautogui.easeInOutQuad)
        time.sleep(0.1)
        
        print("🖱️ Clicking card...")
        pyautogui.click()
        time.sleep(0.2)
        
        if return_to_start:
            print(f"🖱️ Returning mouse to original position: ({start_x}, {start_y})")
            pyautogui.moveTo(start_x, start_y, duration=move_duration, tween=pyautogui.easeInOutQuad)

if __name__ == "__main__":
    try:
        ctrl = BridgeController()
        ctrl.click_build_info()
    except Exception as e:
        print(f"Controller action failed: {e}")
