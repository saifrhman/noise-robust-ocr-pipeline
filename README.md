# 🧾 Noise-Robust OCR Pipeline

A practical OCR pipeline designed to improve text recognition on noisy, real-world receipt images.

This project explores how image preprocessing techniques impact OCR performance and implements an adaptive selection strategy to improve robustness.

---

## 🚀 Features

- Receipt OCR using EasyOCR
- Image preprocessing (CLAHE, denoise, thresholding)
- Adaptive **auto-mode** selection
- OCR confidence scoring
- Character accuracy evaluation (SROIE v2)
- Streamlit demo application
- Text cleaning / normalization
- Basic receipt field extraction:
  - Merchant (heuristic)
  - Date
  - Total candidates

---

## 🧠 Motivation

OCR on receipts is notoriously unreliable due to:

- Low contrast
- Blur
- Compression artifacts
- Small fonts
- Shadows / lighting variations

Instead of blindly applying preprocessing, this project evaluates:

> *When preprocessing helps vs hurts OCR performance.*

---

## 📊 Evaluation (SROIE v2 Dataset)

Evaluation performed on **200 training** and **200 test** receipts.

### **Train Split**
- Baseline (grayscale): **0.1215**
- Auto-mode OCR: **0.1284**
- Improvement: **+0.0069**
- Improved samples: **56 / 200**

### **Test Split**
- Baseline (grayscale): **0.1285**
- Auto-mode OCR: **0.1359**
- Improvement: **+0.0074**
- Improved samples: **53 / 200**

Auto-mode adaptively selects preprocessing strategies based on:

- OCR confidence
- Heuristic text-quality scoring
- Margin-based switching

---

## 🖥 Demo Application

Interactive Streamlit-based **Receipt OCR Extractor**

### Capabilities

- Upload receipt images
- Adaptive OCR preprocessing (auto / manual modes)
- OCR confidence + scoring
- Cleaned OCR output
- Raw OCR inspection
- Merchant / date / total extraction
- JSON export
- CSV session history

---

## ⚙️ Run Locally

### 1️⃣ Clone repo

```bash
git clone https://github.com/saifrhman/noise-robust-ocr-pipeline.git
cd noise-robust-ocr-pipeline


## Results (SROIE v2 – Receipt OCR)

Evaluation performed on 200 training and 200 test receipts.

### Train Split
- Baseline (grayscale): 0.1215
- Auto-mode OCR: 0.1284
- Improvement: +0.0069
- Improved samples: 56 / 200

### Test Split
- Baseline (grayscale): 0.1285
- Auto-mode OCR: 0.1359
- Improvement: +0.0074
- Improved samples: 53 / 200

Auto-mode adaptively selects preprocessing strategies (none / CLAHE / denoise)
based on OCR confidence and text-quality scoring.
