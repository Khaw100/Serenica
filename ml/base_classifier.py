"""
ml/base_classifier.py
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


class BaseClassifier(ABC):

    LABEL_NAMES: list[str] = []  # override di subclass jika perlu

    def __init__(
        self,
        model_dir: str,
        threshold: Optional[float] = None,
        max_length: int = 128,
        device: Optional[str] = None,
    ):
        self.model_dir  = Path(model_dir)
        self.max_length = max_length

        # ── Device ──────────────────────────────────────────────────────────
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info(f"[{self.__class__.__name__}] Device: {self.device}")

        # ── Load tokenizer & model ───────────────────────────────────────────
        logger.info(f"[{self.__class__.__name__}] Loading from: {self.model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model     = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        self.model.to(self.device)
        self.model.eval()

        # ── Label names: prioritas dari model.config.id2label ───────────────
        # Kalau id2label sudah ada di config.json (hasil save_pretrained),
        # pakai itu. Kalau tidak, fallback ke LABEL_NAMES subclass.
        if self.model.config.id2label and len(self.model.config.id2label) > 1:
            self.label_names = [
                self.model.config.id2label[i]
                for i in range(len(self.model.config.id2label))
            ]
            logger.info(f"[{self.__class__.__name__}] Labels loaded from model config ({len(self.label_names)} labels)")
        elif self.LABEL_NAMES:
            self.label_names = self.LABEL_NAMES
            logger.info(f"[{self.__class__.__name__}] Labels loaded from class ({len(self.label_names)} labels)")
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Tidak ada label names! "
                f"Pastikan model config.json punya id2label, atau set LABEL_NAMES di subclass."
            )

        # ── Threshold ────────────────────────────────────────────────────────
        self.threshold = threshold if threshold is not None else self._load_threshold()
        logger.info(f"[{self.__class__.__name__}] Threshold: {self.threshold} ✅")

    def _load_threshold(self) -> float:
        """Load threshold dari emotion_config.pt atau model_config.pt."""
        for fname in ["emotion_config.pt", "model_config.pt"]:
            config_path = self.model_dir / fname
            if config_path.exists():
                cfg = torch.load(str(config_path), map_location="cpu")
                thr = float(cfg.get("threshold", 0.3))
                logger.info(f"[{self.__class__.__name__}] Threshold {thr} dari {fname}")
                return thr
        logger.warning(f"[{self.__class__.__name__}] Config tidak ditemukan, pakai default 0.3")
        return 0.3

    # ── Shared inference ─────────────────────────────────────────────────────

    def _tokenize(self, texts: list[str]) -> dict:
        return self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        )

    def _run_model(self, inputs: dict) -> torch.Tensor:
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return torch.sigmoid(outputs.logits).cpu()

    def _to_result_dict(
        self,
        probs: torch.Tensor,
        threshold: float,
        return_all: bool = False,
    ) -> dict[str, float]:
        if return_all:
            return {
                self.label_names[i]: round(float(probs[i]), 4)
                for i in range(len(self.label_names))
            }

        result = {
            self.label_names[i]: round(float(probs[i]), 4)
            for i in range(len(self.label_names))
            if probs[i] >= threshold
        }

        # fallback: kalau tidak ada yang lolos threshold → ambil top-1
        if not result:
            top_idx = int(torch.argmax(probs))
            result  = {self.label_names[top_idx]: round(float(probs[top_idx]), 4)}

        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    # ── Public API ───────────────────────────────────────────────────────────

    @abstractmethod
    def predict(self, text: str, **kwargs) -> dict:
        ...

    def predict_batch(
        self,
        texts: list[str],
        threshold: Optional[float] = None,
        return_all: bool = False,
    ) -> list[dict[str, float]]:
        thr    = threshold if threshold is not None else self.threshold
        inputs = self._tokenize(texts)
        probs  = self._run_model(inputs)
        return [self._to_result_dict(row, thr, return_all) for row in probs]

    def get_top_k(self, text: str, k: int = 3) -> dict[str, float]:
        all_scores = self.predict(text, return_all=True)
        return dict(sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:k])

    def set_threshold(self, threshold: float) -> None:
        self.threshold = threshold
        logger.info(f"[{self.__class__.__name__}] Threshold updated → {self.threshold}")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  model_dir  = '{self.model_dir}'\n"
            f"  threshold  = {self.threshold}\n"
            f"  device     = '{self.device}'\n"
            f"  num_labels = {len(self.label_names)}\n"
            f"  labels     = {self.label_names[:5]}...\n"
            f")"
        )