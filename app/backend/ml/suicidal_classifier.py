import re
import logging
import joblib
from pathlib import Path
from typing import Optional

import nltk
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)


class SuicidalClassifier:
    LABEL_NAMES = ["non-suicidal", "suicidal"]

    def __init__(
        self,
        model_path: str = "./models/suicidal_classifier/suicide_detection_pipeline.pkl",
        threshold: float = 0.8,
        spacy_model: str = "en_core_web_sm",
        batch_size: int = 64,
    ):
        self.model_path = Path(model_path)
        self.threshold  = threshold
        self.batch_size = batch_size

        # Load sklearn pipeline
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"[SuicidalClassifier] Model tidak ditemukan: {self.model_path}\n"
                f"Pastikan file suicide_detection_pipeline.pkl ada di path tersebut."
            )

        logger.info(f"[SuicidalClassifier] Loading from: {self.model_path}")
        self.pipeline = joblib.load(str(self.model_path))

        # Setup NLTK stopwords
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)

        self._stop_words = set(stopwords.words("english"))
        self._protected  = {
            "i", "me", "my", "myself",
            "you", "your", "yours",
            "he", "him", "his", "himself",
            "she", "her", "hers", "herself",
            "they", "them", "their", "theirs",
        }
        self._filtered_stopwords = self._stop_words - self._protected

        # Load spaCy untuk lemmatization
        self.nlp = self._load_spacy(spacy_model)

        logger.info(f"[SuicidalClassifier] Threshold: {self.threshold} | Ready!")

    def _load_spacy(self, model_name: str):
        """Load spaCy model, auto-download kalau belum ada."""
        import spacy
        try:
            nlp = spacy.load(model_name, disable=["parser", "ner"])
            logger.info(f"[SuicidalClassifier] spaCy '{model_name}' loaded")
            return nlp
        except OSError:
            logger.warning(f"[SuicidalClassifier] Downloading spaCy '{model_name}'...")
            from spacy.cli import download
            download(model_name)
            import spacy
            return spacy.load(model_name, disable=["parser", "ner"])

    def _selective_stopword_removal(self, text: str) -> str:
        words = text.split()
        return " ".join(w for w in words if w not in self._filtered_stopwords)

    def _clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _lemmatize_single(self, text: str) -> str:
        doc = self.nlp(text)
        tokens = [
            token.text.lower() if token.pos_ == "PRON"
            else token.lemma_.lower()
            for token in doc
        ]
        return " ".join(tokens)

    def _lemmatize_batch(self, texts: list[str]) -> list[str]:
        results = []
        for doc in self.nlp.pipe(texts, batch_size=self.batch_size):
            tokens = [
                token.text.lower() if token.pos_ == "PRON"
                else token.lemma_.lower()
                for token in doc
            ]
            results.append(" ".join(tokens))
        return results

    def _preprocess(self, text: str) -> str:
        text = self._selective_stopword_removal(text)
        text = self._clean_text(text)
        text = self._lemmatize_single(text)
        return text

    def _preprocess_batch(self, texts: list[str]) -> list[str]:
        partially_cleaned = [
            self._clean_text(self._selective_stopword_removal(t))
            for t in texts
        ]
        return self._lemmatize_batch(partially_cleaned)

    def predict(
        self,
        text: str,
        threshold: Optional[float] = None,
    ) -> dict:
        thr     = threshold if threshold is not None else self.threshold
        cleaned = self._preprocess(text)
        proba   = self.pipeline.predict_proba([cleaned])[0]

        suicidal_prob = float(proba[1])

        return {
            "is_suicidal" : suicidal_prob >= thr,
            "confidence"  : round(suicidal_prob, 4),
            "label"       : "suicidal" if suicidal_prob >= thr else "non-suicidal",
            "scores"      : {
                "non-suicidal": round(float(proba[0]), 4),
                "suicidal"    : round(suicidal_prob, 4),
            },
        }

    def predict_batch(
        self,
        texts: list[str],
        threshold: Optional[float] = None,
    ) -> list[dict]:
        thr     = threshold if threshold is not None else self.threshold
        cleaned = self._preprocess_batch(texts)
        probas  = self.pipeline.predict_proba(cleaned)

        results = []
        for proba in probas:
            suicidal_prob = float(proba[1])
            results.append({
                "is_suicidal" : suicidal_prob >= thr,
                "confidence"  : round(suicidal_prob, 4),
                "label"       : "suicidal" if suicidal_prob >= thr else "non-suicidal",
                "scores"      : {
                    "non-suicidal": round(float(proba[0]), 4),
                    "suicidal"    : round(suicidal_prob, 4),
                },
            })
        return results

    def set_threshold(self, threshold: float) -> None:
        self.threshold = threshold
        logger.info(f"[SuicidalClassifier] Threshold updated → {self.threshold}")

    def __repr__(self) -> str:
        return (
            f"SuicidalClassifier(\n"
            f"  model_path = '{self.model_path}'\n"
            f"  threshold  = {self.threshold}\n"
            f"  model_type = sklearn Pipeline (TF-IDF + SGD)\n"
            f"  preprocess = stopword → clean → spaCy lemmatize\n"
            f")"
        )