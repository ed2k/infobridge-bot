import cv2
import numpy as np

def main():
    img = cv2.imread("debug_captures/live_ui_all_sides.png")
    if img is None:
        print("Error")
        return
        
    blocks = [
        (19, 75, "Block 1"),
        (142, 75, "Block 2"),
        (279, 75, "Block 3")
    ]
    
    # We want to inspect the region where the suit symbol resides.
    # In each block, the suit symbol is on the left side, e.g. x_start .. x_start+25, y_start+25 .. y_start+55.
    for bx, by, name in blocks:
        print(f"\n=== {name} (x={bx}, y={by}) ===")
        crop = img[by+25 : by+55, bx+2 : bx+22]
        h, w = crop.shape[:2]
        
        # Calculate color statistics
        # Let's count pixels that are close to red, black, blue, green etc.
        # Print a small grid of the actual color names/channels
        for r in range(0, h, 3):
            row_str = ""
            for c in range(0, w, 3):
                b, g, r_val = crop[r, c]
                # Print color code
                if r_val > 150 and g < 100 and b < 100:
                    row_str += " R " # Red
                elif r_val < 50 and g < 50 and b < 50:
                    row_str += " K " # Black
                elif b > 150 and r_val < 100 and g < 100:
                    row_str += " B " # Blue
                elif g > 150 and r_val < 100 and b < 100:
                    row_str += " G " # Green
                elif r_val > 200 and g > 200 and b > 200:
                    row_str += " W " # White
                else:
                    row_str += f"({r_val:02x})"
            print(row_str)

if __name__ == "__main__":
    main()
