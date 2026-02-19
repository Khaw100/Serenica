"""
ml/mlflow_loader.py

Load classifier dari MLflow Model Registry,
sebagai pengganti load langsung dari folder local.

Cara pakai di aplikasi:
    from ml.mlflow_loader import load_classifiers

    clfs = load_classifiers()
    clfs["emotion"].predict("I feel so happy today")
    clfs["suicidal"].predict("I want to end my life")
    clfs["problem"].predict("I can't stop worrying")
"""

import logging
import os
from functools import lru_cache
from typing import Optional

import mlflow.pyfunc
import pandas as pd

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

# Model names di Registry — harus sama dengan registered_model_name di push script
_REGISTERED_NAMES = {
    "emotion" : "emotion-classifier",
    "suicidal": "suicidal-classifier",
    "problem" : "problem-classifier",
}


# ═════════════════════════════════════════════════════════════════════════════
# Thin wrapper supaya interface-nya sama dengan ProblemClassifier / EmotionClassifier
# ═════════════════════════════════════════════════════════════════════════════

class MLflowClassifierWrapper:
    """
    Wrapper tipis di atas mlflow.pyfunc model.
    Expose interface yang sama dengan BaseClassifier:
        .predict(text)
        .predict_batch(texts)
        .get_top_k(text, k)
    """

    def __init__(self, name: str, model_uri: str):
        self.name = name
        logger.info(f"[MLflow] Loading '{name}' dari {model_uri} ...")
        self._model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"[MLflow] '{name}' loaded ✅")

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

        if return_all:
            return {col: round(float(row[col]), 4) for col in result_df.columns}

        # Filter by threshold, fallback top-1
        filtered = {col: round(float(row[col]), 4)
                    for col in result_df.columns if row[col] >= threshold}
        if not filtered:
            top_col = row.idxmax()
            filtered = {top_col: round(float(row[top_col]), 4)}

        return dict(sorted(filtered.items(), key=lambda x: x[1], reverse=True))

    def predict_batch(
        self,
        texts: list[str],
        threshold: float = 0.3,
        return_all: bool = False,
    ) -> list[dict]:
        result_df = self._call(texts)
        results   = []
        for _, row in result_df.iterrows():
            if return_all:
                results.append({col: round(float(row[col]), 4) for col in result_df.columns})
                continue
            filtered = {col: round(float(row[col]), 4)
                        for col in result_df.columns if row[col] >= threshold}
            if not filtered:
                top_col = row.idxmax()
                filtered = {top_col: round(float(row[top_col]), 4)}
            results.append(dict(sorted(filtered.items(), key=lambda x: x[1], reverse=True)))
        return results

    def get_top_k(self, text: str, k: int = 3) -> dict:
        all_scores = self.predict(text, return_all=True)
        return dict(sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:k])


# ═════════════════════════════════════════════════════════════════════════════
# Suicidal wrapper khusus — outputnya beda (is_suicidal, confidence, label)
# ═════════════════════════════════════════════════════════════════════════════

class MLflowSuicidalWrapper:
    """Wrapper untuk suicidal classifier yang output-nya berbeda."""

    def __init__(self, model_uri: str):
        logger.info(f"[MLflow] Loading 'suicidal' dari {model_uri} ...")
        self._model = mlflow.pyfunc.load_model(model_uri)
        logger.info("[MLflow] 'suicidal' loaded ✅")

    def _call(self, texts: list[str]) -> pd.DataFrame:
        return self._model.predict(pd.DataFrame({"text": texts}))

    def predict(self, text: str) -> dict:
        return self._call([text]).iloc[0].to_dict()

    def predict_batch(self, texts: list[str]) -> list[dict]:
        df = self._call(texts)
        return [row.to_dict() for _, row in df.iterrows()]


# ═════════════════════════════════════════════════════════════════════════════
# Public loader
# ═════════════════════════════════════════════════════════════════════════════

def _model_uri(registered_name: str, stage: str = "latest") -> str:
    """
    Build model URI untuk MLflow Registry.
    stage: 'latest' / 'Production' / 'Staging' / 'None'
    """
    if stage == "latest":
        return f"models:/{registered_name}/latest"
    return f"models:/{registered_name}/{stage}"


@lru_cache(maxsize=1)
def load_classifiers(
    tracking_uri: Optional[str] = None,
    stage: str = "latest",
) -> dict:
    """
    Load semua classifier dari MLflow Registry.
    Di-cache supaya tidak re-load setiap request.

    Parameters
    ----------
    tracking_uri : str, optional
        Override MLflow URI. Default dari env MLFLOW_TRACKING_URI.
    stage : str
        Model stage: 'latest', 'Production', 'Staging'.

    Returns
    -------
    dict
        {
          "emotion" : MLflowClassifierWrapper,
          "suicidal": MLflowSuicidalWrapper,
          "problem" : MLflowClassifierWrapper,
        }

    Examples
    --------
    >>> clfs = load_classifiers()
    >>> clfs["emotion"].predict("I feel so happy today")
    {'joy': 0.91}
    >>> clfs["problem"].predict("I can't stop worrying")
    {'anxiety': 0.87}
    >>> clfs["suicidal"].predict("I want to end my life")
    {'label': 'suicidal', 'confidence': 0.95, 'is_suicidal': True}
    """
    uri = tracking_uri or MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    logger.info(f"[MLflow] Tracking URI: {uri} | Stage: {stage}")

    clfs = {
        "emotion": MLflowClassifierWrapper(
            name      = "emotion",
            model_uri = _model_uri(_REGISTERED_NAMES["emotion"], stage),
        ),
        "suicidal": MLflowSuicidalWrapper(
            model_uri = _model_uri(_REGISTERED_NAMES["suicidal"], stage),
        ),
        "problem": MLflowClassifierWrapper(
            name      = "problem",
            model_uri = _model_uri(_REGISTERED_NAMES["problem"], stage),
        ),
    }

    logger.info("[MLflow] Semua model loaded ✅")
    return clfs