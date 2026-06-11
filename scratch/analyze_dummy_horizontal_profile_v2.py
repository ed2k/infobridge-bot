import cv2
import numpy as np

def main():
    img = cv2.imread("debug_captures/1_ui_full.png")
    if img is None:
        print("Failed to load image")
        return
        
    # Crop: y=275..310 (35 pixels high), x=0..510
    crop = img[275:310, 0:510]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # In BBO, the background is light gray/white.
    # Text/symbols are dark. Let's threshold so that non-background pixels are 1.
    # We can use a simple threshold like < 220 (since gray background is usually > 220).
    thresh = (gray < 220).astype(np.uint8)
    
    # 1D projection: sum of text pixels along the vertical axis for each x
    proj = np.sum(thresh, axis=0)
    
    # Print profile in chunks of 10 pixels to see the layout
    print("Horizontal projection of text pixels (chunked every 10px):")
    for i in range(0, 510, 10):
        chunk = proj[i:i+10]
        # print visual bar
        bar = "".join(["#" if val > 2 else "." for val in chunk])
        print(f"x={i:3d}..{i+9:3d}: {bar} | values={[int(v) for v in chunk]}")

if __name__ == "__main__":
    main()
