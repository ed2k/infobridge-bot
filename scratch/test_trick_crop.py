import cv2

def main():
    img = cv2.imread("debug_captures/1_ui_full.png")
    # Relative coordinates of the yellow square:
    # x_rel = 67, y_rel = 228, w = 235, h = 292
    x, y, w, h = 67, 228, 235, 292
    
    crop = img[y:y+h, x:x+w]
    cv2.imwrite("/Users/admin/.gemini/antigravity-ide/brain/99eec12b-56cc-4eb2-b9c6-920fc66c5416/test_trick_crop.png", crop)
    print("Saved crop.")

if __name__ == "__main__":
    main()
