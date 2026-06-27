import cv2
import os

def main():
    for name in ["1_ui_full.png", "2_bidding.png", "5_bidding_hint.png", "bidding_area.png"]:
        path = os.path.join("debug", name)
        if os.path.exists(path):
            img = cv2.imread(path)
            print(f"{name}: shape = {img.shape}")
        else:
            print(f"{name}: not found")

if __name__ == "__main__":
    main()
