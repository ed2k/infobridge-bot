import cv2
import pytesseract

def clean_rank(text):
    # Mapping for ranks
    mapping = {
        "0": "Q", "O": "Q", "D": "Q",
        "1": "T", "S": "J",
        "N": "T", "W": "T",
        "Z": "",
        "E": "6",
        "M": "", "B": "", "I": "", "F": "", "H": "", "X": "",
    }
    
    cleaned = []
    text_upper = text.strip().upper().replace("\n", " ").replace("10", "T")
    for char in text_upper:
        if char.isdigit():
            cleaned.append(char)
        elif char in mapping:
            repl = mapping[char]
            if repl:
                cleaned.append(repl)
        elif char in ["A", "K", "Q", "J", "T"]:
            cleaned.append(char)
            
    filtered = []
    for c in cleaned:
        if not filtered or filtered[-1] != c:
            filtered.append(c)
            
    # Keep only characters in the valid rank set
    rank_order = {r for r in ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]}
    return [r for r in filtered if r in rank_order]

def test_split(img_path):
    print(f"\n--- Testing {img_path} with column segmentation ---")
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to load image")
        return
        
    dummy_text_crop = img[275:310, 0:510]
    
    # Define the 4 columns based on color transitions
    cols = [
        ("Col1", 0, 166),
        ("Col2", 166, 275),
        ("Col3", 275, 355),
        ("Col4", 355, 510)
    ]
    
    for name, x1, x2 in cols:
        col_crop = dummy_text_crop[:, x1:x2]
        gray = cv2.cvtColor(col_crop, cv2.COLOR_BGR2GRAY)
        
        # Test a couple of fx/threshold parameters
        for fx in [4.0]:
            scaled = cv2.resize(gray, (0, 0), fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
            thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            proc = cv2.bitwise_not(thresh)
            
            try:
                txt = pytesseract.image_to_string(proc, config="--psm 6")
                ranks = clean_rank(txt)
                print(f"  {name} (x={x1}..{x2}): raw='{txt.strip()}' -> ranks={ranks}")
            except Exception as e:
                print(f"  {name} Error: {e}")

if __name__ == "__main__":
    test_split("debug_captures/live_ui_all_sides.png")
    test_split("debug_captures/1_ui_full.png")
