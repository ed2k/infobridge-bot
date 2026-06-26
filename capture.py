#!/usr/bin/env python3
"""
Screen Capture Module for Bridge Bot.
Captures regions of the screen using mss and converts to OpenCV images.
Handles macOS Retina scaling by resizing to match logical points.
"""

import os
import json
import cv2
import numpy as np
import mss
from PIL import Image

CONFIG_FILE = "config.json"

class ScreenCapture:
    def __init__(self, config_path=CONFIG_FILE):
        self.config_path = config_path
        self.config = self.load_config()
        self.sct = mss.mss()

    def close(self):
        """Release the mss resources."""
        if hasattr(self, 'sct') and self.sct:
            try:
                self.sct.close()
            except Exception:
                pass

    def __del__(self):
        self.close()

    def load_config(self):
        if not os.path.exists(self.config_path):
            print("⚠️ config.json not found. Generating default coordinates based on screen size...")
            try:
                import pyautogui
                w, h = pyautogui.size()
            except Exception:
                # Fallback to standard 1440x900 resolution
                w, h = 1440, 900
            
            # Default coordinates targeting the top-right quadrant of the screen
            config = {
                "ui_roi": {
                    "x": int(w * 0.5),
                    "y": 0,
                    "width": int(w * 0.5),
                    "height": int(h * 0.6)
                },
                "bidding_hint_roi": {
                    "x": int(w * 0.55),
                    "y": int(h * 0.05),
                    "width": int(w * 0.25),
                    "height": int(h * 0.1)
                },
                "bidding_roi": {
                    "x": int(w * 0.55),
                    "y": int(h * 0.15),
                    "width": int(w * 0.25),
                    "height": int(h * 0.3)
                },
                "trick_roi": {
                    "x": int(w * 0.55),
                    "y": int(h * 0.15),
                    "width": int(w * 0.25),
                    "height": int(h * 0.3)
                },
                "player_hand_roi": {
                    "x": int(w * 0.52),
                    "y": int(h * 0.45),
                    "width": int(w * 0.35),
                    "height": int(h * 0.12)
                },
                "build_info_btn": {
                    "x": int(w * 0.85),
                    "y": int(h * 0.55)
                }
            }
            with open(self.config_path, "w") as f:
                json.dump(config, f, indent=4)
            print(f"Created default coordinate profile at {self.config_path} for screen size {w}x{h}.")
            
        with open(self.config_path, "r") as f:
            return json.load(f)

    def _get_bounding_roi(self):
        keys = ["bidding_hint_roi", "bidding_roi", "trick_roi", "player_hand_roi", "ui_roi"]
        xs, ys, rights, bottoms = [], [], [], []
        for k in keys:
            r = self.config.get(k)
            if r:
                xs.append(r["x"])
                ys.append(r["y"])
                rights.append(r["x"] + r["width"])
                bottoms.append(r["y"] + r["height"])
        if not xs:
            return {"x": 0, "y": 0, "width": 1, "height": 1}
        return {
            "x": min(xs), "y": min(ys),
            "width": max(rights) - min(xs),
            "height": max(bottoms) - min(ys),
        }

    def capture_game_panel(self):
        """Captures the minimal bounding rect covering all game ROIs in one shot."""
        return self.capture_region(self._get_bounding_roi())

    def crop_from_panel(self, panel_img, roi):
        """Crops a sub-region from the full game panel image."""
        panel_roi = self._get_bounding_roi()
        ox, oy = panel_roi["x"], panel_roi["y"]
        x = roi["x"] - ox
        y = roi["y"] - oy
        return panel_img[y:y + roi["height"], x:x + roi["width"]]

    def capture_region(self, roi):
        """
        Captures a screen region defined by a dict with x, y, width, height.
        Returns an OpenCV BGR image.
        """
        # mss monitor dict format
        monitor = {
            "top": int(roi["y"]),
            "left": int(roi["x"]),
            "width": int(roi["width"]),
            "height": int(roi["height"])
        }

        # Grab screenshot reusing self.sct
        screenshot = self.sct.grab(monitor)
        
        # Convert to numpy array
        img = np.array(screenshot)
        
        # mss returns BGRA, convert to BGR
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # Handle macOS Retina scaling.
        # If the image size doesn't match logical size, resize it.
        h, w = img_bgr.shape[:2]
        target_w = int(roi["width"])
        target_h = int(roi["height"])
        
        if w != target_w or h != target_h:
            img_bgr = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
            
        return img_bgr

    def capture_ui(self):
        """Captures the entire calibrated Bridge play UI."""
        return self.capture_region(self.config["ui_roi"])

    def capture_bidding(self):
        """Captures the bidding history/box region."""
        return self.capture_region(self.config["bidding_roi"])

    def capture_bidding_hint(self):
        """Captures the bidding hint text above the bidding headers."""
        return self.capture_region(self.config["bidding_hint_roi"])

    def capture_trick(self):
        """Captures the trick play area."""
        return self.capture_region(self.config["trick_roi"])

    def capture_player_hand(self):
        """Captures the player's own hand region."""
        return self.capture_region(self.config["player_hand_roi"])

    def save_debug_images(self, output_dir="debug"):
        """Captures all regions and saves them as images for visual debugging."""
        os.makedirs(output_dir, exist_ok=True)
        try:
            ui = self.capture_ui()
            cv2.imwrite(os.path.join(output_dir, "1_ui_full.png"), ui)
            print("Saved full UI capture to debug/1_ui_full.png")
        except Exception as e:
            print(f"Error capturing UI: {e}")

        try:
            bidding = self.capture_bidding()
            cv2.imwrite(os.path.join(output_dir, "2_bidding.png"), bidding)
            print("Saved bidding capture to debug/2_bidding.png")
        except Exception as e:
            print(f"Error capturing bidding: {e}")

        try:
            trick = self.capture_trick()
            cv2.imwrite(os.path.join(output_dir, "3_trick.png"), trick)
            print("Saved trick capture to debug/3_trick.png")
        except Exception as e:
            print(f"Error capturing trick: {e}")

        try:
            hand = self.capture_player_hand()
            cv2.imwrite(os.path.join(output_dir, "4_player_hand.png"), hand)
            print("Saved player hand capture to debug/4_player_hand.png")
        except Exception as e:
            print(f"Error capturing player hand: {e}")

        try:
            hint = self.capture_bidding_hint()
            cv2.imwrite(os.path.join(output_dir, "5_bidding_hint.png"), hint)
            print("Saved bidding hint capture to debug/5_bidding_hint.png")
        except Exception as e:
            print(f"Error capturing bidding hint: {e}")

if __name__ == "__main__":
    try:
        cap = ScreenCapture()
        cap.save_debug_images()
    except Exception as e:
        print(f"Capture failed: {e}")
