import cv2

def main():
    img = cv2.imread("debug_captures/1_ui_full.png")
    # West hand relative coordinates:
    # x = 0 to 68, y = 350 to 780
    x, y, w, h = 0, 350, 68, 430
    
    crop = img[y:y+h, x:x+w]
    cv2.imwrite("/Users/admin/.gemini/antigravity-ide/brain/99eec12b-56cc-4eb2-b9c6-920fc66c5416/west_hand_crop.png", crop)
    print("Saved crop.")

if __name__ == "__main__":
    main()
