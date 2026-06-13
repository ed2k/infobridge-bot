import os
import cv2

def main():
    templates_dir = "templates"
    for filename in sorted(os.listdir(templates_dir)):
        if filename.endswith(".png"):
            path = os.path.join(templates_dir, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            print(f"{filename}: shape={img.shape}")

if __name__ == "__main__":
    main()
