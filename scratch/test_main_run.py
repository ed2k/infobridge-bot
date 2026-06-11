import cv2
import main
from capture import ScreenCapture
import strategy
import time

class MockScreenCapture:
    def __init__(self):
        # Load the mock board image
        self.img = cv2.imread("sample_board.png")
        if self.img is None:
            # Generate if missing
            import generate_mock
            generate_mock.main()
            self.img = cv2.imread("sample_board.png")
            
        h, w = self.img.shape[:2]
        self.config = {
            "ui_roi": {"x": 0, "y": 0, "width": w, "height": h},
            "bidding_hint_roi": {"x": 800, "y": 20, "width": 350, "height": 60},
            "bidding_roi": {"x": 800, "y": 80, "width": 350, "height": 250},
            "trick_roi": {"x": 300, "y": 250, "width": 400, "height": 300},
            "player_hand_roi": {"x": 300, "y": 600, "width": 500, "height": 60}
        }
        
    def capture_game_panel(self):
        return self.img
        
    def crop_from_panel(self, panel_img, roi):
        x, y, w, h = roi["x"], roi["y"], roi["width"], roi["height"]
        return panel_img[y:y+h, x:x+w]
        
    def capture_ui(self):
        return self.img

def main_test():
    # Mock ScreenCapture inside main.py
    main.ScreenCapture = MockScreenCapture
    
    print("Running main.run_decision_loop in once mode...")
    try:
        main.run_decision_loop(interval=1.0, dry_run=True, verbose=True, once=True)
    except Exception as e:
        print(f"Exception raised: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main_test()
