# src/inference/predictor.py

import torch
from .loader import load_emotion_model

_bundle = None  # cache biar ga reload terus

def predict_emotion(text: str, top_k=None):
    global _bundle

    if _bundle is None:
        _bundle = load_emotion_model()

    model = _bundle["model"]
    tokenizer = _bundle["tokenizer"]
    threshold = _bundle["threshold"]
    id2label = _bundle["id2label"]
    device = _bundle["device"]

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits)[0]

    results = {
        id2label[i]: float(probs[i])
        for i in range(len(probs))
        if probs[i] >= threshold
    }

    # optional: sort by confidence
    results = dict(
        sorted(results.items(), key=lambda x: x[1], reverse=True)
    )

    if top_k:
        results = dict(list(results.items())[:top_k])

    return results
