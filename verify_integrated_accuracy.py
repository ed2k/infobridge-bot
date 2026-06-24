import os
import cv2
import re
import numpy as np
import sys

workspace_dir = os.path.dirname(os.path.abspath(__file__))
debug_dir = os.path.join(workspace_dir, "debug")
templates_dir = os.path.join(workspace_dir, "templates")

if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

from analyzer import BridgeAnalyzer

def parse_filename(filename):
    m1 = re.match(r"^card_crop_(?:linear_)?([a-zA-Z0-9]+)_([AKQJT2-9]|None)_?(spade|heart|diamond|club)\.png$", filename)
    if m1: return m1.group(2), m1.group(3)
    m2 = re.match(r"^dummy_card_[a-zA-Z]+_([AKQJT2-9]|None)_?(spade|heart|diamond|club)\.png$", filename)
    if m2: return m2.group(1), m2.group(2)
    m3 = re.match(r"^card_(?:crop_)?([AKQJT2-9]|None)_?(spade|heart|diamond|club)\.png$", filename)
    if m3: return m3.group(1), m3.group(2)
    return None

def main():
    if not os.path.exists(debug_dir) or not os.path.exists(templates_dir):
        print(f"Error: debug or templates directory missing.")
        return

    analyzer = BridgeAnalyzer(templates_dir=templates_dir, verbose=False)
    
    # Load templates
    ranks = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
    for r in ranks:
        analyzer.load_rank_template(r)
        
    # Load debug cases
    files = sorted(os.listdir(debug_dir))
    cases = []
    for f in files:
        if not f.endswith(".png"): continue
        parsed = parse_filename(f)
        if parsed and parsed[0] != "None":
            gt_r, gt_s = parsed
            filepath = os.path.join(debug_dir, f)
            img = cv2.imread(filepath)
            if img is not None and img.shape[0] == 60 and img.shape[1] == 40:
                cases.append({"filename": f, "gt": gt_r, "img": img})
                
    print(f"Evaluating {len(cases)} cases with Integrated Template Matching + PaddleOCR Hybrid Override...")

    correct = 0
    failures = []
    y1, y2, x1, x2 = 2, 38, 2, 36

    for case in cases:
        img = case["img"]
        gt = case["gt"]
        
        # 1. Shave below suit dynamically
        shaved_card = analyzer.shave_below_suit(img)
        gray = cv2.cvtColor(shaved_card, cv2.COLOR_BGR2GRAY)
        crop = gray[y1:y2, x1:x2]
        
        # 2. Shave borders
        final_crop = analyzer.shave_borders(crop)
        
        # 3. Get OCR prediction
        paddle_r = analyzer.paddle_verify_rank(img, initial_rank=None)
        
        # 4. Template matching score calculation
        scores = {}
        for r in ranks:
            scores[r] = analyzer.score_rank_candidate(final_crop, r)
            
        best_cand = max(scores, key=scores.get)
        best_score = scores[best_cand]
        
        local_r = best_cand
        local_score = best_score
        
        chosen_rank = local_r
        
        if paddle_r and paddle_r != local_r:
            paddle_score = scores.get(paddle_r, -1.0)
            
            # Hybrid override rules:
            if local_r == "4" and paddle_r in ["2", "3"] and paddle_score > 0.40:
                chosen_rank = paddle_r
            elif paddle_score > 0.60 and paddle_score > local_score + 0.10:
                chosen_rank = paddle_r
                
        if chosen_rank == gt:
            correct += 1
        else:
            failures.append({
                "filename": case["filename"],
                "gt": gt,
                "pred": chosen_rank,
                "local": local_r,
                "paddle": paddle_r
            })
            
    acc = (correct / len(cases)) * 100
    print("\n" + "=" * 60)
    print(f"INTEGRATED ACCURACY: {acc:.2f}% ({correct}/{len(cases)})")
    print("=" * 60)
    
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for f in failures:
            print(f"  {f['filename']}: GT={f['gt']}, Chosen={f['pred']} (local={f['local']}, paddle={f['paddle']})")
    else:
        print("\nAll cases matched perfectly!")

if __name__ == "__main__":
    main()
