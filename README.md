# 🛣️ Road Pothole Detection from Dashcam Footage Using Deep CNN

[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14.2%2B-black.svg)](https://nextjs.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, production-grade Deep Learning system engineered to detect road surface hazards (potholes vs. normal pavement) from dashcam video feeds and photographic road inspection imagery in real-time. Powered by a two-stage fine-tuned **InceptionV3** transfer learning architecture, a high-throughput **FastAPI** backend, and a modern **Next.js** interactive dashboard.

---

## 🌟 Key Highlights & Performance

* **Test Accuracy:** **98.64%** on held-out unseen test samples (664 images).
* **Pothole Precision:** **100.0%** (Zero false positive alerts — critical for driver trust & road management).
* **Pothole Recall:** **97.28%** (322 out of 331 potholes detected).
* **F1-Score:** **98.62%** | **ROC-AUC:** **0.9998** | **PR-AUC:** **0.9998**.
* **Real-Time Dashcam Video Pipeline:** Processes video frames, applies temporal smoothing, generates bounding alerts, and exports annotated MP4 video feeds.

---

## 📊 Datasets Used

The model was trained, validated, and evaluated on a combined multi-source benchmark:

1. **Atulya Kumar Pothole Dataset** ([Kaggle: `atulyakumar98/pothole-detection-dataset`](https://www.kaggle.com/datasets/atulyakumar98/pothole-detection-dataset)):
   * Primary baseline dataset providing paired `Normal` road surfaces and `Pothole` images (~681 raw images).
2. **Andrew MVD Pothole Dataset** ([Kaggle: `andrewmvd/pothole-detection`](https://www.kaggle.com/datasets/andrewmvd/pothole-detection)):
   * Integrated via `extract_and_integrate_dataset.py` to enrich the distribution with varied cavity geometries, lighting conditions, and roadway perspectives (~665 pothole images).
3. **ImageNet (ILSVRC)**:
   * 1.4 million pre-training images for generalized visual representations (edges, textures, spatial features) embedded in the InceptionV3 backbone.

### Dataset Partitioning (70% Train / 15% Val / 15% Test)

| Partition | Normal Road Images | Pothole Images | Total Images |
| :--- | :---: | :---: | :---: |
| **Training (70%)** | 1,667 | 1,650 | **3,317** |
| **Validation (15%)** | 333 | 330 | **663** |
| **Held-Out Test (15%)** | 333 | 331 | **664** |
| **Total** | **2,333** | **2,311** | **4,644** |

---

## 🧠 Model Architecture & Training Strategy

```
Input Image (299 × 299 × 3)
         │
         ▼
InceptionV3 Feature Extractor (ImageNet Pre-trained)
(Multi-scale factorized convolutions: 1x1, 3x3, 5x5, asymmetric 1x7 & 7x1)
         │
         ▼
GlobalAveragePooling2D (2048-dim feature vector)
         │
         ▼
Dense Layer (256 units, ReLU activation, L2 regularization)
         │
         ▼
Dropout Layer (Rate = 0.40 for regularized generalizability)
         │
         ▼
Dense Output Layer (1 unit, Sigmoid activation -> Pothole Probability [0.0, 1.0])
```

### Two-Stage Transfer Learning:
1. **Stage 1 (Head Training):** The InceptionV3 backbone is frozen. Only the classification head is trained for 15 epochs with Adam optimizer ($\text{lr} = 0.001$).
2. **Stage 2 (Fine-Tuning):** The top 30 convolutional layers of InceptionV3 are unfrozen and fine-tuned for 10 epochs at a low learning rate ($\text{lr} = 10^{-5}$) with `ReduceLROnPlateau` and `EarlyStopping`.

---

## 📁 Repository Structure

```
├── backend/                  # FastAPI inference backend
│   ├── app/
│   │   ├── main.py           # Application entrypoint & CORS config
│   │   ├── config.py         # Backend configuration settings
│   │   ├── model_service.py  # Model loading & single-image inference
│   │   ├── video_service.py  # Dashcam video processing & annotation
│   │   ├── schemas.py        # Pydantic request/response schemas
│   │   └── routes/           # REST endpoints (/predict, /health, /metrics)
│   └── requirements.txt      # Backend Python dependencies
├── frontend/                 # Next.js 14 + Tailwind CSS dashboard
│   ├── app/                  # Next.js App Router (pages & components)
│   ├── package.json          # Node dependencies & npm scripts
│   └── tailwind.config.js    # Visual design tokens & styling
├── src/                      # Core deep learning modules
│   ├── config.py             # Hyperparameters, file paths, seeds
│   ├── model.py              # InceptionV3 model architecture builder
│   ├── data_utils.py         # Deduplication, scanning & stratified splits
│   └── evaluation.py         # Evaluation metrics & visualization plotting
├── data/
│   └── raw/                  # Master deduplicated raw image repository
├── dataset/                  # Stratified train, validation, and test splits
│   ├── train/                # Normal / Pothole
│   ├── validation/           # Normal / Pothole
│   └── test/                 # Normal / Pothole
├── models/                   # Saved model artifacts (.keras & configs)
│   ├── best_model.keras      # Checkpointed model with lowest val_loss
│   └── final_model.keras     # Exported production model
├── results/                  # Training curves, confusion matrices, logs
├── outputs/                  # Video inference exports & sample media
├── train.py                  # Two-stage model training script
├── evaluate.py               # Comprehensive model evaluation script
├── prepare_dataset.py        # Dataset validation & splitting pipeline
├── extract_and_integrate_dataset.py # Multi-dataset merger & deduplication
├── generate_word_report.py   # Automated comprehensive DOCX project report generator
└── requirements.txt          # Python project requirements
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
* **Python:** 3.11+
* **Node.js:** 18+ & **npm**
* **Hardware:** GPU recommended (NVIDIA CUDA) but runs seamlessly on modern CPU.

---

### 2. Environment Setup

#### Python Environment:
```bash
# Clone or navigate to the repository
cd "Deep Learning"

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Frontend Dependencies:
```bash
cd frontend
npm install
cd ..
```

---

### 3. Running the System

#### Start the FastAPI Backend:
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```
* Interactive Swagger Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
* Health Check: [http://localhost:8000/api/health](http://localhost:8000/api/health)

#### Start the Next.js Frontend Dashboard:
```bash
cd frontend
npm run dev
```
* Web Dashboard: [http://localhost:3000](http://localhost:3000)

---

### 4. Running Training & Evaluation

#### Prepare & Split Datasets:
```bash
python prepare_dataset.py
```

#### Train the InceptionV3 Model:
```bash
python train.py
```

#### Evaluate on Held-Out Test Set:
```bash
python evaluate.py
```
* Produces confusion matrix, ROC-AUC curve, and detailed classification reports inside the `results/` directory.

#### Generate Academic Word Report (.docx):
```bash
python generate_word_report.py
```
* Generates a fully formatted, publication-ready project report inside `Project_Report/Deep_Learning_Project_Report.docx`.

---

## 📈 Evaluation Results

| Metric | Score | Details |
|---|:---:|---|
| **Accuracy** | **98.64%** | Correctly classified 655 / 664 test images |
| **Precision (Pothole)** | **100.0%** | Zero false alarms triggered |
| **Recall (Pothole)** | **97.28%** | 322 out of 331 potholes detected |
| **Specificity (Normal)** | **100.0%** | 333 out of 333 normal road surfaces identified |
| **F1-Score** | **98.62%** | Harmonic mean of precision and recall |
| **ROC-AUC** | **0.9998** | High separability across all confidence thresholds |
| **Inference Latency** | **~24 ms/frame** | Ready for real-time edge & dashcam deployment |

---

## 📡 API Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Service status, active model version, and device type (CPU/GPU) |
| `GET` | `/api/metrics` | Returns test accuracy, precision, recall, and confusion matrix |
| `POST` | `/api/predict/image` | Upload image -> returns classification, confidence score, and alert level |
| `POST` | `/api/predict/video` | Upload dashcam video -> processes frames, annotates video, and returns detection telemetry |

---

## 📜 Academic References & Citations

1. **Szegedy, C., et al. (2016).** *"Rethinking the Inception Architecture for Computer Vision."* IEEE CVPR, pp. 2818–2826.
2. **Kumar, A. (2020).** *"Pothole Detection Dataset."* Kaggle. [kaggle.com/datasets/atulyakumar98/pothole-detection-dataset](https://www.kaggle.com/datasets/atulyakumar98/pothole-detection-dataset).
3. **Laraxa / Andrew MVD (2020).** *"Pothole Detection."* Kaggle. [kaggle.com/datasets/andrewmvd/pothole-detection](https://www.kaggle.com/datasets/andrewmvd/pothole-detection).
4. **Deng, J., et al. (2009).** *"ImageNet: A Large-Scale Hierarchical Image Database."* IEEE CVPR.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
