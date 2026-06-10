import cv2
import numpy as np
import csv
from io import StringIO
import pytesseract
from analyzer import BridgeAnalyzer

def main():
    img = cv2.imread("/Users/admin/.gemini/antigravity-ide/brain/99eec12b-56cc-4eb2-b9c6-920fc66c5416/live_bidding_img.png")
    analyzer = BridgeAnalyzer(verbose=True)
    
    fx = 4.0
    processed = analyzer.preprocess_for_ocr(img)
    data_str = pytesseract.image_to_data(processed, config="--psm 6", output_type=pytesseract.Output.STRING)
    
    f = StringIO(data_str)
    reader = csv.reader(f, delimiter='\t')
    header = next(reader)
    
    left_idx = header.index('left')
    top_idx = header.index('top')
    width_idx = header.index('width')
    height_idx = header.index('height')
    text_idx = header.index('text')
    
    for row in reader:
        if len(row) <= text_idx:
            continue
        text = row[text_idx].strip()
        if not text:
            continue
            
        # If the text starts with a digit 1-7, let's look at it
        if len(text) >= 2 and text[0] in "1234567":
            left = int(row[left_idx])
            top = int(row[top_idx])
            width = int(row[width_idx])
            height = int(row[height_idx])
            
            # Map back to original image coordinates
            x1 = int(left / fx)
            y1 = int(top / fx)
            x2 = int((left + width) / fx)
            y2 = int((top + height) / fx)
            
            # Crop the word region
            word_crop = img[y1:y2, x1:x2]
            
            # Save the word crop
            cv2.imwrite(f"/Users/admin/.gemini/antigravity-ide/brain/99eec12b-56cc-4eb2-b9c6-920fc66c5416/word_{text}.png", word_crop)
            
            # The suit is in the right half of the word crop
            # Let's crop the right 60% of the word box
            suit_w = int(word_crop.shape[1] * 0.6)
            suit_crop = word_crop[:, word_crop.shape[1] - suit_w:]
            
            # Save the suit crop
            cv2.imwrite(f"/Users/admin/.gemini/antigravity-ide/brain/99eec12b-56cc-4eb2-b9c6-920fc66c5416/suit_{text}.png", suit_crop)
            
            # Run template matching on the suit crop
            suit = analyzer.classify_suit_template_matching(suit_crop)
            if not suit:
                suit = analyzer.classify_suit_by_color_shape(suit_crop)
                
            print(f"Text: {text} -> Level={text[0]}, Suit Classified={suit}")

if __name__ == "__main__":
    main()
