# 🏎️ F1 2026 Apex Guardian

> **A production-grade MLOps system that ingests live Formula 1 telemetry, detects high-speed power anomalies with an Isolation Forest model, and serves real-time AI diagnostics via a Streamlit dashboard — fully automated with GitHub Actions and AWS S3.**

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face%20Spaces-yellow?style=for-the-badge)](https://huggingface.co/spaces/glen-louis/F1-2026-Apex-Guardian)
[![MLflow on DagsHub](https://img.shields.io/badge/📊%20Experiment%20Tracking-DagsHub%20%2F%20MLflow-blue?style=for-the-badge)](https://dagshub.com)
[![CI/CD Pipeline](https://img.shields.io/badge/🚀%20CI%2FCD-GitHub%20Actions-black?style=for-the-badge&logo=github)](https://github.com/glen-louis/f1_apex_guardian/actions)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

---

## 🧠 What Problem Does This Solve?

In the 2026 Formula 1 season, cars run a **50/50 hybrid power split** between combustion and electric motors. When a car loses speed on a high-speed straight, it's not always a mechanical failure — it could be intentional energy harvesting (**Super Clipping**), active aerodynamics switching (**Z-Mode**), or a driver lifting early to save fuel.

**Apex Guardian** is an AI system that watches every telemetry frame at speeds **above 280 km/h** and classifies *why* the car is losing power — a tool that would sit in a real pit wall engineering stack.

| Diagnostic Label | Meaning |
|---|---|
| 🔴 **SUPER CLIPPING** | Car intentionally harvesting battery energy on a straight (strategy) |
| 🔵 **Z-MODE TRANSITION** | Active aero closing to prepare for a corner (aerodynamics) |
| 📉 **Driver Lift & Coast** | ML caught the driver partially lifting the throttle to save fuel |
| 🔋 **Severe High-Speed Derating** | Critical failure in electrical deployment at 300+ km/h |
| ⚠️ **Unknown Power Loss** | Unclassified anomaly (mechanical or aero issue) |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GitHub Actions (CI/CD)                   │
│             Runs every Monday · Triggered on push           │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    MLOps Pipeline (src/model.py)            │
│                                                             │
│  1. Ingestion ──► FastF1 API → Raw Telemetry DataFrame      │
│  2. Feature Eng ► master_diagnostic() → Physics Features    │
│  3. Feature Store► Parquet → AWS S3                         │
│  4. Drift Check ► Kolmogorov-Smirnov vs. golden baseline    │
│  5. Training ───► Isolation Forest on high-speed data       │
│  6. Challenger ──► New model vs. Champion metric check      │
│  7. Promotion ──► Winner → S3 Production + DagsHub/MLflow   │
└───────────┬────────────────────────────────────────────────-┘
            │
       ┌────┴────┐
       │  AWS S3  │  ← Feature Store + Production Model Registry
       └────┬────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────┐
│               Streamlit Dashboard (app.py)                  │
│                                                             │
│  • Downloads Champion model from S3 on startup             │
│  • Runs full Inference pipeline (process → predict)        │
│  • Renders Speed Trace, Spatial Diagnostic Map, 3D Plot    │
│  • Deployed to Hugging Face Spaces via hf_sync.yml          │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔬 ML Pipeline Deep Dive

### 1. Data Ingestion (`src/ingestion.py`)
- Pulls **live telemetry** via the [FastF1](https://theoehrly.github.io/Fast-F1/) library
- Auto-detects the **most recent completed race** from the 2026 F1 calendar
- Falls back to **Bahrain Pre-Season Testing** data when the season hasn't started
- Extracts the **fastest lap** for every driver and concatenates into a single DataFrame

### 2. Feature Engineering (`src/processing.py`)
Three physics-based detection functions run sequentially:

- **`detect_super_clipping()`** — Flags frames where the car is at 100% throttle (`≥99%`), above 250 km/h, but *losing speed* (smoothed 5-point rolling average acceleration `≤ 0`)
- **`detect_active_aero()`** — Identifies X-Mode (speed gain without RPM increase) vs. Z-Mode (speed drop despite full throttle, entering a corner)
- **`master_diagnostic()`** — Combines both signals into a single `Diagnostic` label using cascading logic rules

### 3. Anomaly Detection — Isolation Forest (`src/model.py`)
- Scoped to **high-speed strait conditions only**: `Speed ≥ 280 km/h` AND `Throttle ≥ 90%`
- Feature set: `[Speed, Throttle, RPM, Acceleration]`
- `contamination=0.01` (1% of high-speed frames are expected to be anomalous)
- Trained with `n_estimators=100`, `random_state=42` for reproducibility
- All parameters and metrics are logged to **DagsHub / MLflow**

### 4. Data Drift Detection (`src/drift_detector.py`)
- Uses the **Kolmogorov-Smirnov test** to compare the current race's feature distributions against a **golden Bahrain baseline**
- Drift is flagged at `p-value < 0.05` — a statistically significant distribution shift
- Results are logged as MLflow metrics (`drift_p_value_Speed`, etc.)

### 5. Challenger Pattern (`src/challenger.py`)
- Before promotion, the newly trained model is benchmarked against the current **S3 Champion**
- A model that finds **zero anomalies** is rejected (likely a training failure)
- The winning model is uploaded to `production/thermal_detector.pkl` on S3

### 6. AI Diagnostic Classifier (`src/maintenance.py`)
Post-inference, each anomaly frame flagged by the Isolation Forest is sub-classified by a rule-based engine:

| Rule | Condition | Label |
|---|---|---|
| 1 | Throttle < 95% | 📉 Driver Lift & Coast |
| 2 | Speed > 290 km/h AND Acceleration < 1.5 | 🔋 Severe High-Speed Derating |
| 3 | Neither | ⚠️ Unknown Power Loss |

---

## 🚀 CI/CD Automation

### `pipeline.yml` — Weekly MLOps Run
```
Trigger: Every Monday at 09:00 UTC (post-race weekend)
         OR manual dispatch from GitHub Actions UI

Steps:
  ├── Checkout repo
  ├── Install uv (Rust-based Python package manager)
  ├── Set up Python 3.12
  ├── pip install from requirements.txt
  └── Run src/model.py (full Ingest → Drift → Train → Promote pipeline)
```
All AWS and DagsHub credentials are injected via **GitHub Secrets** — zero credentials in code.

### `hf_sync.yml` — Hugging Face Deployment
```
Trigger: Every push to main branch

Steps:
  ├── Checkout repo
  ├── Strip local .git history (removes dev artifacts like mlflow.db)
  ├── Init a clean 'main' branch
  └── Force-push to HF Spaces (glen-louis/F1-2026-Apex-Guardian)
```

---

## 📊 Dashboard Features

The live Streamlit app (`app.py`) provides:

| Panel | Description |
|---|---|
| **AI Diagnostic Insights** | Real-time anomaly count with severity alerts |
| **KPI Metrics** | Max speed, harvesting events, ML anomaly count |
| **Speed vs. Distance Trace** | Multi-driver speed trace with Plotly |
| **🗺️ Spatial Diagnostic Map** | Color-coded track map by diagnostic label |
| **🧊 3D AI Decision Space** | Speed × RPM × Acceleration scatter plot showing model decision boundary |
| **Sidebar MLOps Badge** | Live display of Feature Store, Registry, CI/CD, and Model info |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **ML Model** | Scikit-learn Isolation Forest |
| **Data Source** | FastF1 (Official F1 Telemetry API) |
| **Feature Store** | AWS S3 (Parquet) |
| **Experiment Tracking** | DagsHub + MLflow |
| **Drift Detection** | SciPy KS-test |
| **Dashboard** | Streamlit + Plotly |
| **CI/CD** | GitHub Actions |
| **Deployment** | Hugging Face Spaces |
| **Package Manager** | uv (Astral) |
| **Python** | 3.12 |

---

## ⚙️ Local Setup

### Prerequisites
- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) package manager
- AWS credentials (for S3 feature store)
- DagsHub token (for MLflow tracking)

### 1. Clone & Install
```bash
git clone https://github.com/glen-louis/f1_apex_guardian.git
cd f1_apex_guardian

uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Fill in your credentials:
# AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME
# DAGSHUB_REPO_OWNER, DAGSHUB_REPO_NAME, DAGSHUB_TOKEN
```

### 3. Run the MLOps Pipeline
```bash
# Ingest → Drift Check → Train → Promote Champion to S3
python src/model.py
```

### 4. Run the Dashboard
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
f1_apex_guardian/
├── .github/
│   ├── README.md               # This file (rendered on GitHub)
│   └── workflows/
│       ├── pipeline.yml        # Weekly automated MLOps pipeline
│       └── hf_sync.yml         # Auto-deploy to Hugging Face Spaces
├── src/
│   ├── ingestion.py            # FastF1 telemetry ingestion
│   ├── processing.py           # Physics-based feature engineering
│   ├── model.py                # Training, drift detection, promotion
│   ├── maintenance.py          # Inference + diagnostic classification
│   ├── drift_detector.py       # KS-test data drift monitor
│   ├── challenger.py           # Challenger vs. Champion evaluation
│   ├── s3_manager.py           # AWS S3 upload/download interface
│   └── automation.py           # Race calendar utility
├── data/                       # Local feature cache (gitignored)
├── models/                     # Local model cache (gitignored)
├── app.py                      # Streamlit dashboard entry point
├── pyproject.toml              # Project metadata & dependencies
└── requirements.txt            # Pinned dependency list
```

---

## 🔑 Required GitHub Secrets

| Secret | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | AWS IAM key for S3 access |
| `AWS_SECRET_ACCESS_KEY` | AWS IAM secret |
| `S3_BUCKET_NAME` | Target S3 bucket name |
| `DAGSHUB_REPO_OWNER` | Your DagsHub username |
| `DAGSHUB_REPO_NAME` | DagsHub repository name |
| `DAGSHUB_TOKEN` | DagsHub personal access token |
| `HF_TOKEN` | Hugging Face write token (for Space deployment) |

---

*Built with 🏎️ by [glen-louis](https://github.com/glen-louis)*
