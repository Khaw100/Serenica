"""
ml/problem_classifier.py

Problem classifier berbasis RoBERTa, extend BaseClassifier.
Model disimpan via save_pretrained + model_config.pt dari notebook training.

Cara pakai:
    from ml.problem_classifier import ProblemClassifier

    clf = ProblemClassifier("models/problem_classifier")
    clf.predict("I can't stop worrying, my heart races all the time")
    # → {'anxiety': 0.87}
"""

import logging
from typing import Optional

import torch

from .base_classifier import BaseClassifier

logger = logging.getLogger(__name__)


class ProblemClassifier(BaseClassifier):

    # Fallback label names (alphabetical, sesuai LabelEncoder.classes_).
    # Akan di-override otomatis oleh id2label dari config.json jika ada.
    LABEL_NAMES = [
        "anxiety",
        "burnout",
        "depression",
        "grief",
        "none",
        "relationship_issues",
        "work_stress",
    ]

    def __init__(
        self,
        model_dir: str = "models/problem_classifier",
        threshold: Optional[float] = None,
        max_length: int = 128,
        device: Optional[str] = None,
    ):
        super().__init__(
            model_dir=model_dir,
            threshold=threshold,
            max_length=max_length,
            device=device,
        )

    def _load_threshold(self) -> float:
        """Override: cari problem_config.pt dulu, fallback ke model_config.pt."""
        for fname in ["problem_config.pt", "model_config.pt"]:
            config_path = self.model_dir / fname
            if config_path.exists():
                cfg = torch.load(str(config_path), map_location="cpu")
                thr = float(cfg.get("threshold", 0.3))
                logger.info(f"[ProblemClassifier] Threshold {thr} dari {fname}")
                return thr
        logger.warning("[ProblemClassifier] Config tidak ditemukan, pakai default 0.3")
        return 0.3

    def predict(
        self,
        text: str,
        threshold: Optional[float] = None,
        return_all: bool = False,
    ) -> dict[str, float]:
        """
        Prediksi kategori masalah dari satu teks.

        Parameters
        ----------
        text : str
            Teks input (jurnal, curhat, dsb).
        threshold : float, optional
            Override threshold untuk call ini saja.
        return_all : bool
            Jika True, kembalikan semua label beserta score-nya.

        Returns
        -------
        dict
            Label → confidence. Hanya label ≥ threshold (atau top-1 fallback).

        Examples
        --------
        >>> clf.predict("I feel completely overwhelmed at work")
        {'burnout': 0.82}

        >>> clf.predict("I feel overwhelmed", return_all=True)
        {'burnout': 0.82, 'anxiety': 0.61, 'work_stress': 0.44, ...}
        """
        thr = threshold if threshold is not None else self.threshold

        # RoBERTa adalah single-label classifier → softmax lebih tepat dari sigmoid.
        # Base._run_model pakai sigmoid (cocok untuk multilabel), jadi kita
        # tokenize + forward sendiri di sini supaya softmax.
        inputs = self._tokenize([text])
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu()[0]  # (num_labels,)

        return self._to_result_dict(probs, thr, return_all)