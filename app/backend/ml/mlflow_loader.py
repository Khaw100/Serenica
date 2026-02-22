import logging
import os
import time
from functools import lru_cache
from typing import Optional

import mlflow.pyfunc
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_WAIT_TIMEOUT = int(os.getenv("MLFLOW_WAIT_TIMEOUT", "60"))

_REGISTERED_NAMES = {
    "emotion": "emotion-classifier",
    "suicidal": "suicidal-classifier",
    "problem": "problem-classifier",
}

# ============================================
# MLflow Connection Helper
# ============================================

def wait_for_mlflow(tracking_uri: str, timeout: int = 60) -> bool:
    """
    Wait for MLflow server to be ready and accepting connections
    
    Args:
        tracking_uri: MLflow tracking URI
        timeout: Max seconds to wait
    
    Returns:
        True if MLflow ready, False otherwise
    """
    import requests
    
    base_url = tracking_uri.replace('/api/', '/').rstrip('/')
    health_url = f"{base_url}/health"
    
    start_time = time.time()
    attempt = 0
    
    while (time.time() - start_time) < timeout:
        attempt += 1
        try:
            logger.info(f"[MLflow] Checking connection (attempt {attempt})...")
            response = requests.get(health_url, timeout=5)
            
            if response.status_code == 200:
                logger.info(f"✅ MLflow server ready at {tracking_uri}")
                
                # Additional check: Try to list experiments
                mlflow.set_tracking_uri(tracking_uri)
                client = MlflowClient()
                client.search_experiments()
                
                logger.info("✅ MLflow API responding correctly")
                return True
                
        except requests.exceptions.ConnectionError as e:
            logger.debug(f"Connection error: {e}")
        except Exception as e:
            logger.debug(f"MLflow not ready: {e}")
        
        time.sleep(2)
    
    logger.error(f"❌ MLflow not ready after {timeout}s")
    return False

# ============================================
# MLflow Wrappers
# ============================================

class MLflowClassifierWrapper:
    def __init__(self, name: str, model_uri: str):
        self.name = name
        logger.info(f"[MLflow] Loading '{name}' from {model_uri} ...")
        self._model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"✅ [MLflow] '{name}' loaded successfully")

    def _call(self, texts: list[str]) -> pd.DataFrame:
        df = pd.DataFrame({"text": texts})
        return self._model.predict(df)

    def predict(
        self,
        text: str,
        threshold: float = 0.3,
        return_all: bool = False,
    ) -> dict:
        result_df = self._call([text])
        row = result_df.iloc[0]
        all_scores = dict(
            sorted(
                {col: round(float(row[col]), 4) for col in result_df.columns}.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        if self.name == "problem":
            return all_scores

        if return_all:
            return all_scores

        filtered = {k: v for k, v in all_scores.items() if v >= threshold}
        if not filtered:
            top = next(iter(all_scores))
            filtered = {top: all_scores[top]}
        return filtered

    def predict_batch(self, texts, threshold=0.3, return_all=False):
        result_df = self._call(texts)
        results = []
        for _, row in result_df.iterrows():
            all_scores = dict(sorted(
                {col: round(float(row[col]), 4) for col in result_df.columns}.items(),
                key=lambda x: x[1], reverse=True,
            ))
            if self.name == "problem" or return_all:
                results.append(all_scores)
                continue
            filtered = {k: v for k, v in all_scores.items() if v >= threshold}
            if not filtered:
                top = next(iter(all_scores))
                filtered = {top: all_scores[top]}
            results.append(filtered)
        return results

    def get_top_k(self, text: str, k: int = 3) -> dict:
        all_scores = self.predict(text, return_all=True)
        return dict(sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:k])


class MLflowSuicidalWrapper:
    def __init__(self, model_uri: str):
        logger.info(f"[MLflow] Loading 'suicidal' from {model_uri} ...")
        self._model = mlflow.pyfunc.load_model(model_uri)
        logger.info("✅ [MLflow] 'suicidal' loaded successfully")

    def _call(self, texts: list[str]) -> pd.DataFrame:
        return self._model.predict(pd.DataFrame({"text": texts}))

    def predict(self, text: str) -> dict:
        return self._call([text]).iloc[0].to_dict()

    def predict_batch(self, texts: list[str]) -> list[dict]:
        df = self._call(texts)
        return [row.to_dict() for _, row in df.iterrows()]


# ============================================
# Model URI Helper
# ============================================

def _model_uri(registered_name: str, stage: str = "latest") -> str:
    """Get model URI for MLflow registry"""
    if stage == "latest":
        return f"models:/{registered_name}/latest"
    return f"models:/{registered_name}/{stage}"


def verify_model_exists(client: MlflowClient, registered_name: str) -> bool:
    """Check if model exists in MLflow registry"""
    try:
        versions = client.search_model_versions(f"name='{registered_name}'")
        if not versions:
            logger.error(f"❌ No versions found for model '{registered_name}'")
            return False
        
        logger.info(f"✅ Found {len(versions)} version(s) for '{registered_name}'")
        return True
    except Exception as e:
        logger.error(f"❌ Error checking model '{registered_name}': {e}")
        return False


# ============================================
# Main Loader (MLflow Only)
# ============================================

@lru_cache(maxsize=1)
def load_classifiers(
    tracking_uri: Optional[str] = None,
    stage: str = "latest",
) -> dict:
    """
    Load classifiers from MLflow registry ONLY
    
    Args:
        tracking_uri: MLflow tracking URI (None = use env)
        stage: Model stage (latest/Production/Staging)
    
    Returns:
        Dict of {model_name: classifier}
    
    Raises:
        RuntimeError: If MLflow is not accessible or models not found
    """
    uri = tracking_uri or MLFLOW_TRACKING_URI
    
    logger.info("="*80)
    logger.info(f"[MLflow] Starting model loading from {uri}")
    logger.info(f"[MLflow] Stage: {stage}")
    logger.info("="*80)
    
    # Wait for MLflow to be ready
    if not wait_for_mlflow(uri, timeout=MLFLOW_WAIT_TIMEOUT):
        raise RuntimeError(f"❌ MLflow server not accessible at {uri}")
    
    # Set tracking URI
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    
    # Verify all models exist before loading
    logger.info("\n[MLflow] Verifying models in registry...")
    for name, registered_name in _REGISTERED_NAMES.items():
        if not verify_model_exists(client, registered_name):
            raise RuntimeError(f"❌ Model '{registered_name}' not found in MLflow registry")
    
    # Load models
    clfs = {}
    logger.info("\n[MLflow] Loading models...")
    
    for name, registered_name in _REGISTERED_NAMES.items():
        try:
            model_uri = _model_uri(registered_name, stage)
            
            if name == "suicidal":
                clfs[name] = MLflowSuicidalWrapper(model_uri)
            else:
                clfs[name] = MLflowClassifierWrapper(name, model_uri)
            
            logger.info(f"✅ Successfully loaded '{name}'")
            
        except MlflowException as e:
            logger.error(f"❌ MLflow error loading '{name}': {e}")
            raise RuntimeError(f"Failed to load '{name}' from MLflow: {e}")
        except Exception as e:
            logger.error(f"❌ Error loading '{name}': {e}")
            raise RuntimeError(f"Failed to load '{name}': {e}")
    
    logger.info("\n" + "="*80)
    logger.info(f"✅ ALL MODELS LOADED FROM MLFLOW")
    logger.info(f"✅ Loaded {len(clfs)}/{len(_REGISTERED_NAMES)} models: {list(clfs.keys())}")
    logger.info("="*80)
    
    return clfs