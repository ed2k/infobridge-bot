import cv2
import os

def main():
    templates_dir = "templates"
    for suit in ["spade", "heart", "diamond", "club"]:
        path = os.path.join(templates_dir, f"{suit}.png")
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            print(f"Template '{suit}': shape {img.shape}")
        else:
            print(f"Template '{suit}': NOT FOUND")

if __name__ == "__main__":
    main()
