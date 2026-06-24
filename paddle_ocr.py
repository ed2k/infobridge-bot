#!/usr/bin/env python3
"""
PaddleOCR wrapper for Bridge Bot.
Provides a unified interface for OCR using PaddleOCR v6 with ONNX runtime.
Falls back gracefully if PaddleOCR is not available.
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple

_paddle_ocr_instance = None


def _get_paddle_ocr():
    """Lazy-initialize PaddleOCR singleton."""
    global _paddle_ocr_instance
    if _paddle_ocr_instance is None:
        try:
            from paddleocr import PaddleOCR
            _paddle_ocr_instance = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang="en",
                engine="onnxruntime",
            )
        except ImportError:
            raise ImportError(
                "PaddleOCR is not installed. Install with: "
                "pip install paddleocr paddlepaddle onnxruntime"
            )
    return _paddle_ocr_instance


def ocr_image(
    img: np.ndarray,
    engine: str = "onnxruntime",
) -> List[Dict]:
    """
    Run PaddleOCR on a BGR image.

    Args:
        img: OpenCV BGR image.
        engine: Inference engine ('onnxruntime' recommended for macOS).

    Returns:
        List of dicts, each with keys:
          - 'text': recognized text string
          - 'confidence': float confidence score
          - 'bbox': list of 4 [x, y] corner points (original image coords)
    """
    if img is None or img.size == 0:
        return []

    ocr = _get_paddle_ocr()
    results = ocr.predict(img)

    detections = []
    for res in results:
        if not res or "rec_texts" not in res:
            continue
        texts = res["rec_texts"]
        scores = res.get("rec_scores", [0.0] * len(texts))
        polys = res.get("rec_polys", [])

        for i, text in enumerate(texts):
            if not text or not text.strip():
                continue
            score = float(scores[i]) if i < len(scores) else 0.0
            bbox = polys[i].tolist() if i < len(polys) else []
            detections.append({
                "text": text.strip(),
                "confidence": score,
                "bbox": bbox,
            })

    return detections


def ocr_text(
    img: np.ndarray,
    min_confidence: float = 0.5,
) -> str:
    """
    Run PaddleOCR and return all recognized text as a single string,
    sorted top-to-bottom, left-to-right.

    Args:
        img: OpenCV BGR image.
        min_confidence: Minimum confidence threshold.

    Returns:
        Concatenated recognized text.
    """
    detections = ocr_image(img)
    filtered = [d for d in detections if d["confidence"] >= min_confidence]

    if not filtered:
        return ""

    def sort_key(d):
        bbox = d["bbox"]
        if len(bbox) >= 1:
            cy = sum(p[1] for p in bbox) / len(bbox)
            cx = sum(p[0] for p in bbox) / len(bbox)
            return (cy // 20, cx)
        return (0, 0)

    filtered.sort(key=sort_key)
    return " ".join(d["text"] for d in filtered)


def ocr_with_positions(
    img: np.ndarray,
    min_confidence: float = 0.3,
) -> List[Tuple[str, float, float, float, float]]:
    """
    Run PaddleOCR and return text with bounding box centers.

    Returns:
        List of (text, center_x, center_y, width, height) tuples.
    """
    detections = ocr_image(img)
    results = []

    for d in detections:
        if d["confidence"] < min_confidence:
            continue
        bbox = d["bbox"]
        if len(bbox) < 4:
            continue

        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w = x_max - x_min
        h = y_max - y_min

        results.append((d["text"], cx, cy, w, h))

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python paddle_ocr.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read image: {img_path}")
        sys.exit(1)

    print(f"Running PaddleOCR on: {img_path}")
    results = ocr_image(img)

    for r in results:
        print(f"  [{r['confidence']:.3f}] {r['text']}")
