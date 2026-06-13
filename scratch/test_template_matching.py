import cv2
import numpy as np
import os

def nms(boxes, threshold=0.5):
    """Non-Maximum Suppression to filter overlapping detections."""
    if len(boxes) == 0:
        return []
    
    # boxes format: (x, y, w, h, score, label)
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    pick = []
    
    while len(boxes) > 0:
        b = boxes[0]
        pick.append(b)
        boxes = boxes[1:]
        
        remaining = []
        for box in boxes:
            # Calculate IoU
            x1 = max(b[0], box[0])
            y1 = max(b[1], box[1])
            x2 = min(b[0] + b[2], box[0] + box[2])
            y2 = min(b[1] + b[3], box[1] + box[3])
            
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            intersection = w * h
            
            area_b = b[2] * b[3]
            area_box = box[2] * box[3]
            union = area_b + area_box - intersection
            
            iou = intersection / union if union > 0 else 0
            if iou < threshold:
                remaining.append(box)
        boxes = remaining
        
    return pick

def main():
    east_path = "debug/dummy_strip_east.png"
    if not os.path.exists(east_path):
        print(f"❌ {east_path} not found.")
        return

    img = cv2.imread(east_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Load all rank templates
    templates_dir = "templates"
    ranks = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
    rank_templates = {}
    for r in ranks:
        p = os.path.join(templates_dir, f"rank_{r}.png")
        if os.path.exists(p):
            rank_templates[r] = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        else:
            p2 = os.path.join(templates_dir, f"{r}.png")
            if os.path.exists(p2):
                rank_templates[r] = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)

    # Load suit templates
    suits = ["spade", "heart", "diamond", "club"]
    suit_templates = {}
    for s in suits:
        p = os.path.join(templates_dir, f"{s}.png")
        if os.path.exists(p):
            suit_templates[s] = cv2.imread(p, cv2.IMREAD_GRAYSCALE)

    print(f"Loaded {len(rank_templates)} rank templates and {len(suit_templates)} suit templates.")

    # Match ranks
    rank_detections = []
    for r, tpl in rank_templates.items():
        res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
        h, w = tpl.shape
        threshold = 0.65
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            score = res[pt[1], pt[0]]
            rank_detections.append((pt[0], pt[1], w, h, score, r))
            
    filtered_ranks = nms(rank_detections, threshold=0.3)
    filtered_ranks.sort(key=lambda x: (x[1], x[0]))

    print("\nRank Detections (Template Matching):")
    for b in filtered_ranks:
        print(f"  Rank: {b[5]:<2} | score: {b[4]:.3f} | x: {b[0]:<3} | y: {b[1]:<3} | w: {b[2]:<2} | h: {b[3]:<2}")

    # Match suits
    suit_detections = []
    for s, tpl in suit_templates.items():
        res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
        h, w = tpl.shape
        threshold = 0.65
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            score = res[pt[1], pt[0]]
            suit_detections.append((pt[0], pt[1], w, h, score, s))

    filtered_suits = nms(suit_detections, threshold=0.3)
    filtered_suits.sort(key=lambda x: (x[1], x[0]))

    print("\nSuit Detections (Template Matching):")
    for b in filtered_suits:
        print(f"  Suit: {b[5]:<8} | score: {b[4]:.3f} | x: {b[0]:<3} | y: {b[1]:<3} | w: {b[2]:<2} | h: {b[3]:<2}")

if __name__ == "__main__":
    main()
