import cv2
import numpy as np

def main():
    img = cv2.imread("debug/dummy_strip_east.png")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    black_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 60, 100]))
    
    # Sum black pixels vertically in row 4 (y=210..250)
    col_sums = np.sum(black_mask[210:250, :] > 0, axis=0)
    
    print("Black pixel column sums in Row 4 (y=210..250):")
    for x in range(len(col_sums)):
        if col_sums[x] > 0:
            print(f"  x={x}: sum={col_sums[x]}")

if __name__ == "__main__":
    main()
