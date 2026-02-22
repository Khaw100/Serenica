import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.backend.ml.mlflow_loader import load_classifiers

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "api.log")

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    fh = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

setup_logging()
logger = logging.getLogger(__name__)

class PredictRequest(BaseModel):
    model: str
    text: str
    threshold: Optional[float] = None
    return_all: Optional[bool] = False

class PredictBatchRequest(BaseModel):
    model: str
    texts: List[str]
    threshold: Optional[float] = None
    return_all: Optional[bool] = False

app = FastAPI(title="Serenica ML API - MLflow Backend")

# Global classifier store
CLFS = {}

@app.on_event("startup")
async def startup_event():
    """Load models from MLflow at startup"""
    global CLFS
    
    logger.info("="*80)
    logger.info("🚀 STARTING SERENICA ML API")
    logger.info("="*80)
    
    try:
        # This will raise RuntimeError if MLflow is not accessible
        CLFS = load_classifiers()
        logger.info(f"✅ API ready with models: {list(CLFS.keys())}")
        
    except RuntimeError as e:
        logger.error(f"❌ CRITICAL: {e}")
        logger.error("❌ API will start but models are NOT available")
        logger.error("❌ Check MLflow server and model registry")
        # Let app start but endpoints will fail
    except Exception as e:
        logger.exception("❌ Unexpected error during startup")

@app.get("/health")
def health():
    """Health check endpoint"""
    status = "ok" if CLFS else "error"
    return {
        "status": status,
        "models_loaded": list(CLFS.keys()),
        "model_count": len(CLFS),
        "mlflow_uri": os.getenv("MLFLOW_TRACKING_URI", "not set")
    }

@app.get("/models")
def list_models():
    """List available models"""
    if not CLFS:
        raise HTTPException(status_code=503, detail="No models loaded")
    return {"models": list(CLFS.keys())}

def _call_predict(clf: Any, text: str, threshold: Optional[float], return_all: bool) -> Dict[str, Any]:
    """Call predict with proper parameter handling"""
    try:
        if threshold is None:
            return clf.predict(text, return_all=return_all)
        return clf.predict(text, threshold=threshold, return_all=return_all)
    except TypeError:
        return clf.predict(text)

@app.post("/predict")
def predict(req: PredictRequest):
    """Single text prediction"""
    if not CLFS:
        raise HTTPException(status_code=503, detail="Models not loaded. Check MLflow connection.")
    
    if req.model not in CLFS:
        available = list(CLFS.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Model '{req.model}' not found. Available: {available}"
        )

    clf = CLFS[req.model]
    logger.info(f"Predict: model={req.model} text_len={len(req.text)}")
    
    try:
        result = _call_predict(clf, req.text, req.threshold, bool(req.return_all))
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))

    return {"model": req.model, "input": req.text, "result": result}

@app.post("/predict_batch")
def predict_batch(req: PredictBatchRequest):
    """Batch prediction"""
    if not CLFS:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if req.model not in CLFS:
        raise HTTPException(status_code=404, detail=f"Model '{req.model}' not found")

    clf = CLFS[req.model]
    logger.info(f"Batch predict: model={req.model} n={len(req.texts)}")
    
    try:
        if hasattr(clf, "predict_batch"):
            results = clf.predict_batch(req.texts, threshold=req.threshold, return_all=bool(req.return_all))
        else:
            results = [_call_predict(clf, t, req.threshold, bool(req.return_all)) for t in req.texts]
    except TypeError:
        results = [clf.predict(t) for t in req.texts]
    except Exception as e:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=str(e))

    return {"model": req.model, "results": results}