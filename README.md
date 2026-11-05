# 🌙 Serenica

**Serenica** is an AI-powered mental health companion that helps users reflect on their emotions, understand their mental state, and connect with the right support. It uses custom-trained machine learning models for multi-label text classification and an OpenAI-powered chat companion — all served through a FastAPI backend and a Streamlit frontend.

![Serenica companion](https://terrarium.rakhawiratama.com/api/creature.svg?user=Khaw100&repo=Serenica)

---

## 🧠 Features

- **Emotion Classification** — Detects primary emotions from free-text journaling (anxiety, depression, stress, burnout, grief, etc.)
- **Problem Classification** — Identifies underlying concerns and themes
- **Suicidal Intent Detection** — Risk screening with crisis alert messaging
- **AI Chat** — Context-aware mental health chat powered by OpenAI (uses emotion analysis results as context)
- **Psychologist Matching** — Recommends specialists based on emotion profile and risk level
- **MLflow Integration** — Models are versioned and served via MLflow Model Registry

---

## 🗂️ Project Structure

```
finalProject/
├── app/
│   ├── backend/          # FastAPI server (port 8000) + ML inference via MLflow
│   │   ├── server.py     # API endpoints: /health, /models, /predict, /predict_batch
│   │   ├── ml/           # MLflow model loaders and classifiers
│   │   └── Dockerfile
│   └── frontend/         # Streamlit app (port 8501)
│       ├── serenica.py   # Main UI: Home, Results, Chat tabs
│       ├── utils/        # API client, session state, chat client, psychologists
│       └── Dockerfile
├── src/
│   └── emotion_project/  # Model training utilities (dataset, builder, trainer)
├── models/               # ⚠️ Gitignored — local model artifacts only
├── mlruns/               # ⚠️ Gitignored — MLflow tracking data
├── mlartifacts/          # ⚠️ Gitignored — MLflow artifact store
├── docker-compose.yml    # Orchestrates mlflow + backend + frontend
├── push_to_mlflow.py     # Script to register trained models to MLflow
├── find_threshold.py     # Script to find optimal classification thresholds
└── requirements.txt      # Root-level dependencies
```

> **Note:** `models/`, `mlruns/`, `mlartifacts/`, and `backend-dummy/` are excluded from version control via `.gitignore`. You must generate or obtain these locally before running.

---

## 🚀 Running the App

### Option 1 — Docker Compose (Recommended)

Runs all three services (MLflow, Backend, Frontend) together.

```powershell
docker-compose up --build
```

| Service  | URL                    |
|----------|------------------------|
| Frontend | http://localhost:8501  |
| Backend  | http://localhost:8000  |
| MLflow   | http://localhost:5000  |

> The backend requires models to be registered in MLflow before it can serve predictions. See [Pushing Models to MLflow](#-pushing-models-to-mlflow) below.

---

### Option 2 — Run Locally (No Docker)

**1. Create and activate a virtual environment:**

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
```

**2. Install dependencies:**

```powershell
# Root-level (training/MLflow tools)
pip install -r requirements.txt

# Backend
pip install -r app/backend/requirements.txt

# Frontend
pip install -r app/frontend/requirements.txt
```

**3. Start the MLflow tracking server:**

```powershell
mlflow server `
  --backend-store-uri ./mlruns `
  --artifacts-destination ./mlartifacts `
  --host 0.0.0.0 `
  --port 5000
```

**4. Start the backend API:**

```powershell
# From the root directory
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
uvicorn app.backend.server:app --host 0.0.0.0 --port 8000
```

**5. Start the frontend:**

```powershell
# From app/frontend/
streamlit run serenica.py --server.port 8501
```

---

## 📦 Pushing Models to MLflow

After training, register your models to the MLflow registry so the backend can load them:

```powershell
python push_to_mlflow.py
```

This registers the three classifiers (`emotion`, `problem`, `suicidal`) under the MLflow Model Registry.

---

## 🎯 Finding Classification Thresholds

To tune prediction confidence thresholds for each classifier:

```powershell
python find_threshold.py
```

---

## 🧪 Training Models

Training scripts live in `src/emotion_project/`:

```powershell
python src/emotion_project/train.py
```

Key modules:
- `dataset_builder.py` — Prepares training data
- `model_builder.py` — Defines model architecture
- `trainer_utils.py` — Training loop utilities
- `config.py` — Hyperparameters and paths
- `labels.py` — Label definitions for each classifier

---

## 🔗 API Endpoints

| Method | Endpoint        | Description                        |
|--------|-----------------|------------------------------------|
| GET    | `/health`       | Health check + loaded model list   |
| GET    | `/models`       | List available models              |
| POST   | `/predict`      | Single text prediction             |
| POST   | `/predict_batch`| Batch text prediction              |

**Example request:**

```json
POST /predict
{
  "model": "emotion",
  "text": "I've been feeling really overwhelmed lately.",
  "threshold": 0.4,
  "return_all": true
}
```

---

## ⚙️ Environment Variables

| Variable              | Default               | Description                          |
|-----------------------|-----------------------|--------------------------------------|
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000`  | MLflow server URI (used by backend)  |
| `BACKEND_URL`         | `http://backend:8000` | Backend URL (used by frontend)       |
| `MLFLOW_WAIT_TIMEOUT` | `90`                  | Seconds to wait for MLflow on startup|

---

## 💬 Chat Feature

The **Chat** tab requires an OpenAI API key entered directly in the UI. The AI companion automatically uses your emotion analysis results as context for personalized, empathetic responses in Bahasa Indonesia.
