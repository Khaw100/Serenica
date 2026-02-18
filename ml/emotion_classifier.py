"""
ml/emotion_classifier.py
"""

from typing import Optional
from .base_classifier import BaseClassifier


class EmotionClassifier(BaseClassifier):
    """
    Multi-label emotion classifier (GoEmotions, 28 kelas).
    Model: RoBERTa fine-tuned pada GoEmotions dataset.

    Label di-load otomatis dari model config.json (id2label).
    Kalau tidak ada, fallback ke LABEL_NAMES di bawah.
    """

    # Fallback — dipakai kalau id2label tidak ada di config.json
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
        """
        Prediksi emosi dari satu teks.

        Args:
            text       : teks input
            threshold  : override threshold (opsional)
            return_all : kalau True, kembalikan semua label + skornya

        Returns:
            dict {emotion: probability} yang melewati threshold,
            diurutkan dari probabilitas tertinggi.

        Example:
            >>> clf = EmotionClassifier("./models/emotion_classifier")
            >>> clf.predict("I feel so happy today!")
            {'joy': 0.9231, 'excitement': 0.7102, 'optimism': 0.5843}
        """
        thr    = threshold if threshold is not None else self.threshold
        inputs = self._tokenize([text])
        probs  = self._run_model(inputs)[0]   # shape: (28,)
        return self._to_result_dict(probs, thr, return_all)