# Screen Scraping & State Verification Implementation Plan

Deterministic coordinate-mapped screen scraping coupled with a Finite State Machine (FSM) verification layer ensures near-100% reliability for reading structured computer screens (e.g., Contract Bridge clients or Bridge Base Online), bypassing the latency and instability of deep learning (YOLO/CNNs).

---

## 1. System Setup & Coordinate-Mapped ROIs
* **Environment Isolation:** Run the game client in a virtualized guest VM and the vision engine on the host. This avoids anti-automation detection and controls visual parameters.
* **Display Settings:** Disable DPI scaling and subpixel font smoothing (e.g., ClearType) to prevent edge-blurring.
* **Anchor & Map ROIs:** Find the window's top-left pixel, then use relative pixel offsets to extract key Regions of Interest: South Hand, Trick Play Matrix (center), Bidding Box, Vulnerability Panel, and Dealer Indicator.

## 2. Low-Level Image Preprocessing
* **HSV Color Isolation:** Convert RGB to HSV color space to isolate suit colors using localized masks, ignoring lighting/brightness changes.
* **Sauvola Thresholding:** Apply Sauvola adaptive local thresholding optimized with integral sum images for $O(1)$ per-pixel text binarization.
* **Bilinear Upscaling:** Upscale small text ROIs by 300% using bilinear interpolation before binarization to sharpen subpixel shapes.

## 3. Dual-Tier Card Recognition
Reduce the search space from 52 cards to 17 assets by recognizing rank (13) and suit (4) independently.
* **Perceptual Hashing (pHashing):** Convert static, unoccluded hand cards to grayscale and match their 2D Discrete Cosine Transform (DCT) 64-bit phash against a lookup dictionary.
* **Template Matching (NCC):** For overlapping or rotated cards in the trick area, apply Normalized Cross-Correlation (NCC) template matching on active symbol bounding boxes.

## 4. Bidding & Text Parsing
* **Character-Level Template Matching:** Extract the 38 legal bids (e.g., 1♣, 7NT, PASS, DOUBLE) directly using character templates to avoid generic OCR (e.g., Tesseract) confusion (e.g., "1" vs "I").
* **Fuzzy Dictionary Search:** Use OCR for unstructured text (like player names) and apply the SymSpell algorithm to correct errors against a dictionary.

## 5. Rule-Based Guard FSM
Wrap vision outputs in a Contract Bridge rule-based FSM to catch visual anomalies and transient noise:
* **Card Conservation Constraint:** Verify that hand cards plus played tricks equals 13. Re-scan on mismatch.
* **Bidding Monotonicity:** Reject out-of-order bids and fallback to alternative template matches with the next highest correlation.
* **Suit-Following Rule:** Track player hand history. Flag and re-evaluate plays violating suit-following rules when historical tracking shows the player still holds the led suit.