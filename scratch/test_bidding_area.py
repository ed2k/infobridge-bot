import cv2
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analyzer import BridgeAnalyzer

def main():
    img_path = "debug/bidding_area.png"
    if not os.path.exists(img_path):
        print(f"Error: {img_path} does not exist.")
        return

    img = cv2.imread(img_path)
    print(f"Loaded bidding image with shape: {img.shape}")
    
    analyzer = BridgeAnalyzer(verbose=True)
    
    print("\n--- Testing Tesseract Bid Extraction ---")
    try:
        tess_results = analyzer._extract_bids_structured(img)
        print("Tesseract results:", tess_results)
    except Exception as e:
        print("Tesseract error:", e)
        
    print("\n--- Testing PaddleOCR Bid Extraction ---")
    try:
        paddle_results = analyzer._extract_bids_paddle(img)
        print("PaddleOCR results:", paddle_results)
    except Exception as e:
        print("PaddleOCR error:", e)

if __name__ == "__main__":
    main()
