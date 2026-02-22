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
    LABEL_NAMES =  ['anxiety', 'burnout', 'depression', 'grief', 'neutral', 'relationship', 'stress']

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

    def predict(
    self,
    text: str,
    threshold: Optional[float] = None, 
    return_all: bool = False,
    ) -> dict[str, float]:
        inputs = self._tokenize([text])
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        probs = torch.softmax(logits, dim=-1).cpu()[0]

        if return_all:
            return {
                label: round(float(probs[i]), 4)
                for i, label in enumerate(self.LABEL_NAMES)
            }
        top_idx = probs.argmax().item()
        return {self.LABEL_NAMES[top_idx]: round(float(probs[top_idx]), 4)}