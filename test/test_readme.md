# Data report — usage & troubleshooting (VSCode Jupyter)

This document explains how to prepare data and run `data_report.ipynb` using VSCode's built-in Jupyter support, export the report, and inspect the separation steps (2.1.1 → 2.1.4).

---

## 0. Collecting data

Before running this notebook, you must collect data **on-farm** following the official procedure. This ensures the recordings and manual measurements are usable for analysis here. Please read and follow the **[test guide](https://docs.google.com/document/d/13REj5iQ-Jfeg0Pcx3C6EWj9jE8QgDL8Iv6JpqjGidCI/edit?pli=1&tab=t.0)**.

Minimum checklist (summary):
- **Set up hardware** per the guide (camera placement, stable power, correct resolution/FPS).
- **AprilTag placement** (if used): no glare; bottom tag exactly **30 cm** above ground.
- **Calibrate** (depth or AprilTag) as instructed before recording cattle passes.
- **Record runs** with smooth framerate; avoid extreme lighting/occlusion where possible.
- **Capture ground truth**: take **tape measurements** and/or clear photos against the tape for each cattle, and keep their order consistent.
- **Export files**:
  - `record.csv` from the algorithm run (includes `timestamp_iso`, `hip_height`, optional `confidence`, `frame_id`).
  - `actual.csv` with manual tape heights (one row per cow in order of appearance).

Only after this step should you proceed to Section 1 (Prerequisites) and run the notebook.

## 1. Prerequisites

- Python 3.8+  
- Recommended: a virtual environment

Install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Put your data files

Place the required files under the repository `data/` folder:
- `data/actual.csv` — manual tape measurements
- `data/record.csv` — auto-generated detection records

Required columns (exact header names are flexible; the notebook normalizes common synonyms):
- actual.csv: contains `hip_height` (and optional `cattle_id` or `order`)
- record.csv: contains `timestamp` / `timestamp_iso`, `hip_height`, optional `confidence`, optional `frame_id`

Example:
```csv
timestamp_iso,frame_id,hip_height,confidence
2025-10-20T18:21:22Z,123,125.4,0.87
```

---

## 3. How to run the notebook (use VSCode Jupyter)

Use Visual Studio Code with the Jupyter / Python extensions:

1. Install VSCode extensions:
   - Python (Microsoft)
   - Jupyter (Microsoft)

2. Open the project folder in VSCode.

3. Open `test/data_report.ipynb` in VSCode. The notebook UI appears inline (top-right will show kernel).

4. Select the Python interpreter for the notebook:
   - Click the kernel selector (top-right of the notebook) and choose the interpreter inside your `.venv` so dependencies match.

5. Run the notebook:
   - Use the "Run All Cells" button (the ▶▶ icon) in the notebook toolbar, or run cellsstep-by-step with the play buttons.

Note: The notebook contains a central configuration cell near the top (constants). Edit those constants (confidence threshold, NEW_CATTLE_FRAME, etc.) and rerun the notebook when needed.

---

## 4. Separation steps (2.1.*) — order is fixed

Separation runs in chronological order:
1. 2.1.1 — Auto separators using `NEW_CATTLE_AUTO` markers
2. 2.1.2 — Manual separators using `NEW_CATTLE_MANUAL` markers
3. 2.1.3 — Timestamp-gap based separator (chooses threshold to match expected count)
4. 2.1.4 — Manual frame marks (last resort): set `NEW_CATTLE_FRAME = [150, 450, ...]` in the central constants cell

How to inspect which step failed:
- Each 2.1.* cell prints a success/failure message (e.g. `Auto separator success` or `Manual frame separator not accurate`). Check notebook output for these messages to see which step succeeded.
- If a step fails, open its output cells to view printed diagnostics (counts detected, chosen thresholds, candidate IDs).
- For manual frame separation (2.1.4), populate `NEW_CATTLE_FRAME` in the central config cell and re-run the notebook.

---

## 5. Export the report (exact commands — do not change)

Export to HTML:
```bash
jupyter nbconvert data_report.ipynb --to html --no-input
```

Export to PDF (webpdf):
```bash
jupyter nbconvert data_report.ipynb --to webpdf --no-input --allow-chromium-download
```

---

## 6. Troubleshooting tips

- If separation fails at an earlier 2.1.1-2.1.4 step, inspect cell outputs and printed diagnostics. Try adjusting:
  - `NEW_CATTLE_FRAME` for manual frame splits (2.1.4)
- Keep `DISPLAY_DEBUG_TABLE = True` while debugging so intermediate tables are visible in VSCode.
- If plots or tables do not render, ensure the notebook kernel is the `.venv` interpreter and run the top cells again.

---

## 7. Important Note: data quality & test procedure
If results are unsatisfactory, this is often due to problems with the input data (record.csv failing to separate cattle correctly) but can also stem from deeper causes such as an algorithm bug or an incorrect field test setup. Before spending time debugging the notebook, verify that data collection followed the official on‑farm test procedure: [test guide](https://docs.google.com/document/d/13REj5iQ-Jfeg0Pcx3C6EWj9jE8QgDL8Iv6JpqjGidCI/edit?pli=1&tab=t.0)

Quick checklist
- Confirm CSVs contain required headers and sample rows look reasonable (timestamps, frame_id, hip_height, confidence).
- Ensure timestamps are monotonic and frame_id values match camera output.
- Verify camera placement, lighting, and test protocol were followed on the farm per the linked guide.
- Try adjusting CONFIDENCE_THRESHOLD or supplying NEW_CATTLE_FRAME for manual splits. If problems persist, re-collect data following the guide and/or open an issue with representative CSV snippets and diagnostics.

Following the test guide and validating raw data first will save time and help distinguish data-collection issues from algorithm faults.