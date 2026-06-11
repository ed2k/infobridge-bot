import cv2
import numpy as np
import time
from analyzer import BridgeAnalyzer
from main import detect_dummy_hands

def test_no_dummy():
    analyzer = BridgeAnalyzer(verbose=True)
    # Create a blank black image resembling a state with no dummy cards
    img = np.zeros((830, 510, 3), dtype=np.uint8)
    
    print("Starting detect_dummy_hands on a blank screen...")
    t0 = time.time()
    res = detect_dummy_hands(img, analyzer)
    t1 = time.time()
    
    print(f"Result: {res}")
    print(f"Time taken: {t1 - t0:.2f} seconds")

if __name__ == "__main__":
    test_no_dummy()
