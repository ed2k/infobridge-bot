#!/usr/bin/env python3
"""
Bootstrap Card Rank Templates.
Extracts verified rank templates (A, K, Q, J, T, 9, 8, 7, 6, 5, 4, 3, 2) from existing card crops
in the debug folder. Uses multi-parameter Tesseract OCR consensus checking to guarantee correctness.
"""

import os
import cv2
import re
import numpy as np
from collections import Counter
from analyzer import BridgeAnalyzer

def get_consensus_ocr(img, analyzer):
    """
    Runs Tesseract OCR using multiple crop boundaries, scaling factors,
    and PSM modes to return a confident consensus character.
    """
    votes = []
    
    # Crop candidates
    crops = {
        "narrow": img[9:35, 4:18],
        "mid": img[9:35, 2:20],
        "wide": img[4:35, 2:22]
    }
    
    for name, crop in crops.items():
        if crop.size == 0:
            continue
        for fx in [3.0, 4.0, 5.0]:
            proc = analyzer.preprocess_for_ocr(crop, fx=fx)
            for psm in [8, 10, 6]:
                config = f"--psm {psm} -c tessedit_char_whitelist=AKQJT1098765432"
                try:
                    import pytesseract
                    txt = pytesseract.image_to_string(proc, config=config).strip().upper().replace("10", "T")
                    if txt == "1":
                        txt = "T"
                    if txt and txt in "AKQJT98765432":
                        votes.append(txt)
                except Exception:
                    pass
                    
    if votes:
        c = Counter(votes)
        consensus, count = c.most_common(1)[0]
        if count >= 3:
            return consensus, count
    return None, 0

def main():
    debug_dir = "debug"
    templates_dir = "templates"
    
    if not os.path.exists(debug_dir):
        print(f"❌ Error: {debug_dir} directory not found.")
        return
        
    os.makedirs(templates_dir, exist_ok=True)
    
    analyzer = BridgeAnalyzer(verbose=False)
    
    # Match filenames like debug/card_crop_0_3heart.png or card_crop_linear_1_Adiamond.png
    pattern = re.compile(r"card_crop_(?:linear_)?\d+_([AKQJT98765432])([a-z]+)\.png")
    
    # Store candidates for each rank
    candidates = {}
    
    print("🔍 Analyzing debug card crops with OCR verification...")
    
    files = sorted(os.listdir(debug_dir))
    for filename in files:
        match = pattern.match(filename)
        if not match:
            continue
        expected_rank = match.group(1)
        suit = match.group(2)
        filepath = os.path.join(debug_dir, filename)
        
        img = cv2.imread(filepath)
        if img is None:
            continue
            
        # Get consensus OCR rank and its vote count
        ocr_rank, vote_count = get_consensus_ocr(img, analyzer)
        
        # Verify correctness
        if ocr_rank != expected_rank:
            # Skip unverified crops
            continue
            
        # Extract rank crop [9:35, 2:20] for standard (26, 18) template shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rank_crop = gray[9:35, 2:20]
        
        # Quality score: We want high standard deviation (high contrast)
        # but we also heavily weight OCR votes and penalize border/black-edge bleeding.
        has_border = np.any(rank_crop[:, 0] < 50) or np.any(rank_crop[:, 1] < 50)
        std_val = np.std(rank_crop)
        score = vote_count * 15 + std_val
        if has_border:
            score -= 100
        
        if expected_rank not in candidates or score > candidates[expected_rank]["score"]:
            candidates[expected_rank] = {
                "filepath": filepath,
                "crop": rank_crop,
                "score": score
            }

    print("\n✂️ Extracting verified rank templates to templates/...")
    
    all_ranks = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
    extracted_count = 0
    
    for rank in all_ranks:
        if rank in candidates:
            info = candidates[rank]
            out_path_rank = os.path.join(templates_dir, f"rank_{rank}.png")
            out_path_short = os.path.join(templates_dir, f"{rank}.png")
            
            cv2.imwrite(out_path_rank, info["crop"])
            cv2.imwrite(out_path_short, info["crop"])
            
            print(f"   Saved rank_{rank}.png & {rank}.png (verified from {os.path.basename(info['filepath'])}, contrast = {info['score']:.2f})")
            extracted_count += 1
        else:
            print(f"   ⚠️ Rank '{rank}' could not be verified in any debug card crop.")
            
    print(f"\n🎉 Successfully extracted {extracted_count}/{len(all_ranks)} verified rank templates!")

if __name__ == "__main__":
    main()
