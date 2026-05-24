import cv2
import os

def main():
    ui_img_path = "/Users/admin/Documents/GitHub/infobridge-bot/debug_captures/1_ui_full.png"
    if not os.path.exists(ui_img_path):
        print("Error: 1_ui_full.png not found")
        return
        
    img = cv2.imread(ui_img_path)
    grid_img = img.copy()
    
    # We will draw horizontal lines from y=100 to y=300 every 5 pixels
    # And label them every 10 pixels on the left (x=50) and right (x=350)
    for y in range(100, 300, 5):
        color = (0, 0, 255) if y % 10 == 0 else (0, 255, 0)
        thickness = 2 if y % 10 == 0 else 1
        cv2.line(grid_img, (50, y), (380, y), color, thickness)
        
        if y % 10 == 0:
            cv2.putText(grid_img, str(y), (10, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            cv2.putText(grid_img, str(y), (390, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            
    # Draw vertical lines for x from 50 to 380 every 10 pixels
    for x in range(50, 380, 10):
        color = (255, 0, 0) if x % 50 == 0 else (255, 255, 0)
        thickness = 2 if x % 50 == 0 else 1
        cv2.line(grid_img, (x, 100), (x, 300), color, thickness)
        if x % 50 == 0:
            cv2.putText(grid_img, str(x), (x - 10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
            
    out_path = "/Users/admin/Documents/GitHub/infobridge-bot/debug_test_crops/ui_grid.png"
    cv2.imwrite(out_path, grid_img)
    print(f"Saved grid image to {out_path}")

if __name__ == "__main__":
    main()
