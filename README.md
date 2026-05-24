# 🃏 Bridge Play UI Automation Bot

A Python-based utility that captures the screen, parses bridge play elements (cards, bids, etc.) using Computer Vision (OpenCV) and OCR (PyTesseract), and performs mouse interaction (moving smoothly and clicking the bid text for Build Info.

---

## 🚀 Features

- **High-Performance Capture:** Uses `mss` for lightning-fast screen capture.
- **Mac Retina Display Compatibility:** Auto-scales pixel outputs to logical coordinates to ensure flawless regional cropping.
- **Interactive Calibration:** Quick CLI calibration script to set coordinates for any monitor size/resolution.
- **Smart CV / OCR Pipeline:**
  - **Bidding History OCR:** Automatically runs text recognition and standardizes bridge calls.
  - **Color & Shape-Based Card Detection:** Deciphers suit shapes (♠, ♥, ♦, ♣) and colors even without templates.
  - **Template Matching Support:** Ready to match custom card graphics/fonts using standard matching templates.
- **Polite Mouse Controller:** Smoothly glides the cursor to click "Build Info" and returns it back to where you were working.

---

## 🛠️ Prerequisites & Installation

### 1. Install System Dependencies
This bot uses **Tesseract OCR** to read bids and card text.

On macOS, install it using [Homebrew](https://brew.sh/):
```bash
brew install tesseract
```

### 2. Configure macOS Permissions
Because this script captures the screen and controls the mouse, macOS requires permissions:
- **Screen Recording:** Grant permissions to your terminal or IDE (e.g., Terminal, iTerm2, VS Code).
- **Accessibility:** Grant permissions to allow PyAutoGUI to simulate mouse clicks.

### 3. Install Python Packages
Clone this repository and install requirements:
```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

---

## 📖 Usage Guide

The bot is operated entirely via a single terminal entrypoint `main.py`.

### Step 1: Calibrate Screen Coordinates
Run the calibration tool to define where the bridge elements are located on your screen:
```bash
.venv/bin/python main.py --calibrate
```
Follow the CLI instructions:
1. Hover your cursor over the requested corner/button.
2. Press **Enter** in the terminal to save the location.
This generates a `config.json` containing your settings.

### Step 2: Verify Calibrations (Optional but Recommended)
Test your coordinates by capturing crop files:
```bash
python3 main.py --capture-debug
```
This saves screen crops to the `debug_captures/` folder. Inspect them to ensure they align nicely with the table, bidding history, and your cards.

### Step 3: Run Single Screen Analysis
Analyze the screen immediately to print currently visible bids and cards:
```bash
python3 main.py --analyze
```

### Step 4: Automate Mouse Interaction
Smoothly move the mouse and click the "Build Info" button:
```bash
python3 main.py --click
```

### Step 5: Continuous Board Monitoring
Monitor the table continuously, printing changes as cards are played or new bids are made:
```bash
python3 main.py --monitor --interval 1.5
```

### Step 6: Run Decision Engine Once
Run one capture/analyze/decide cycle and exit (instead of looping forever):
```bash
.venv/bin/python main.py --run --once
```

---

## 🔧 Project Structure

- `main.py` - Core CLI orchestrator.
- `calibrate.py` - Terminal-based UI region calibration.
- `capture.py` - Screen region grabber.
- `analyzer.py` - OCR text cleaner and card classification logic.
- `controller.py` - PyAutoGUI mouse automation hooks.
- `requirements.txt` - Python module list.
- `templates/` - Optional folder where you can drop custom template images (`spade.png`, `heart.png`, `diamond.png`, `club.png`) for precision template matching.
