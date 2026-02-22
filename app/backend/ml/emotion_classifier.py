"""
ml/emotion_classifier.py
"""

from typing import Optional
from .base_classifier import BaseClassifier


class EmotionClassifier(BaseClassifier):
    LABEL_NAMES = [
        "admiration", "amusement", "anger", "annoyance", "approval",
        "caring", "confusion", "curiosity", "desire", "disappointment",
        "disapproval", "disgust", "embarrassment", "excitement", "fear",
        "gratitude", "grief", "joy", "love", "nervousness",
        "optimism", "pride", "realization", "relief", "remorse",
        "sadness", "surprise", "neutral"
    ]

    def predict(
        self,
        text: str,
        threshold: Optional[float] = None,
        return_all: bool = False,
    ) -> dict[str, float]:
        thr    = threshold if threshold is not None else self.threshold
        inputs = self._tokenize([text])
        probs  = self._run_model(inputs)[0]
        return self._to_result_dict(probs, thr, return_all)