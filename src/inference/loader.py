import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "models/emotion_models"

def load_emotion_model(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    cfg = torch.load(f"{MODEL_DIR}/emotion_config.pt", map_location=device)

    threshold = cfg.get("threshold", 0.5)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "threshold": threshold,
        "id2label": model.config.id2label,
        "device": device
    }
