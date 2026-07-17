# 🃏 InfoBridge Bot

A Python screen-scraping and automation bot for contract bridge clients. It captures calibrated UI regions, reads bids and cards with OpenCV + OCR + template matching, tracks the deal over time, applies simple rule-based strategy, and can optionally click bids and cards with the mouse.

**Observation-only by default.** Mouse automation is off unless you pass `--action`.

---

## Features

- **Fast screen capture** via `mss`, with Retina/logical coordinate scaling for macOS
- **Interactive calibration** of UI regions (table, bidding, hand, trick, bid buttons) into `config.json`
- **CV / OCR pipeline**
  - Bidding history OCR with standardized bridge calls
  - Rank/suit detection via color, shape, and image templates
  - Dummy hands, trick area, and player hand recognition
- **Game tracker** for bids, hands, and completed tricks; optional export to **PBN** and **JSON**
- **Rule-based strategy** (HCP / suit-length bidding heuristics; follow-suit play helpers)
- **Mouse controller** for smooth clicks when automation is enabled
- **Change-aware monitoring** so static screens are not re-analyzed every poll

---

## Prerequisites & Installation

### 1. System dependencies

Tesseract is used for bid/card text OCR:

```bash
brew install tesseract
```

### 2. macOS permissions

Grant these to your terminal or IDE:

- **Screen Recording** — required for capture
- **Accessibility** — required for PyAutoGUI mouse control

### 3. Python packages

Python 3.11 recommended:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

Or with pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quick start

```bash
# 1. Map UI regions (writes config.json)
.venv/bin/python main.py --calibrate

# 2. Verify crops look correct
.venv/bin/python main.py --capture-debug

# 3. One-shot read of bids and cards
.venv/bin/python main.py --analyze

# 4. Watch the table continuously
.venv/bin/python main.py --monitor --interval 1.5

# 5. Decision loop (logs only; no mouse)
.venv/bin/python main.py --run --once

# 6. Decision loop with real clicks
.venv/bin/python main.py --run --action
```

---

## CLI reference

All commands go through `main.py`. Modes are mutually exclusive.

| Mode | Description |
|------|-------------|
| `--calibrate` | Interactive ROI / button calibration → `config.json` |
| `--capture-debug` | Grab calibrated regions and save crops under `debug/` |
| `--analyze` | Run one OCR/CV analysis pass and print results |
| `--monitor` | Continuously watch for bid/card changes |
| `--run` | Capture → analyze → decide (and optionally act) |
| `--click` | Click the calibrated contract / bid UI |
| `--explain-last-bid` | Click the last bid to read its tooltip explanation |
| `--update-templates` | Refresh rank templates from a live capture |
| `--generate-mock` | Build `debug/sample_board.png` from live or synthetic data |

### Common flags

| Flag | Description |
|------|-------------|
| `--interval N` | Poll interval in seconds (default: `2.0`) |
| `--once` | Run a single iteration then exit (useful with `--run`) |
| `--action` | Enable real mouse movement/clicks (default: observe only) |
| `--dry-run` | Log planned decisions/actions without moving the mouse |
| `--verbose` | Extra CV / template-matching detail |
| `--save-play` | Write captured bids, hand, and play sequence to disk |
| `--output-dir DIR` | Output directory for saved plays (default: `captured_plays`) |

### Examples

```bash
# Continuous monitor, save deals as they complete
.venv/bin/python main.py --monitor --save-play --output-dir captured_plays

# One decision cycle with verbose CV logging
.venv/bin/python main.py --run --once --verbose

# Auto-play loop (observation first, then enable actions when ready)
.venv/bin/python main.py --run --dry-run
.venv/bin/python main.py --run --action --interval 1.5
```

---

## How it works

1. **Calibrate** — hover corners/buttons and press Enter to map regions in `config.json`
2. **Capture** — grab only the configured ROIs (UI, bidding, trick, hand, etc.)
3. **Analyze** — OCR bids; match ranks/suits with templates and color/shape heuristics
4. **Track** — maintain bidding history, initial hands (including dummy), and trick sequence
5. **Decide** — simple strategy for next bid or card
6. **Act** (optional) — smooth mouse moves to click bids or cards when `--action` is set

Vision is coordinate-mapped and rule-checked rather than full deep-learning detection; see `implementation.md` for the design notes.

---

## Project structure

| Path | Role |
|------|------|
| `main.py` | CLI orchestrator (calibrate, analyze, monitor, run) |
| `calibrate.py` | Interactive region calibration |
| `capture.py` | Screen region capture (`mss`) |
| `analyzer.py` | Bid/card analysis, OCR cleanup, classification |
| `detector.py` | Lower-level card/suit detection helpers |
| `tracker.py` | Deal state; PBN/JSON export |
| `strategy.py` | Rule-based bidding and play decisions |
| `controller.py` | PyAutoGUI mouse automation |
| `paddle_ocr.py` / `read_ocr.py` | OCR helpers |
| `bootstrap_templates.py` / `bootstrap_ranks.py` | Template bootstrap utilities |
| `update_templates_live.py` | Live rank-template updates |
| `generate_mock.py` | Sample board image generation |
| `config.json` | Calibrated ROIs and button positions |
| `templates/` | Rank, suit, and bid template images |
| `templates_centered/` | Centered rank templates |
| `tests/` | Unit/integration tests |
| `debug/` | Runtime crops and debug images |
| `scratch/` | Experimental scripts (not part of the main CLI) |
| `implementation.md` | Screen-scraping / FSM design notes |
| `requirements.txt` | Python dependencies |

---

## Testing

```bash
.venv/bin/python -m pytest tests/
```

---

## Notes & limitations

- Tuned for a **desktop bridge client UI** with fixed layout; re-calibrate after resolution or window changes
- Strategy is **heuristic**, not a full double-dummy or system-aware bidding engine
- Automation can mis-click if ROIs drift or recognition is wrong — start with `--analyze`, `--monitor`, and `--dry-run`
- `scratch/` and large `debug/` dumps are for development; they are not required to run the bot
