"""
utils/api_client.py
Handles all communication with the FastAPI backend.
Falls back to mock data if backend is unreachable.
"""

import os
import numpy as np
import requests
import streamlit as st
from datetime import datetime

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def check_backend_health() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ── Mock fallback ────────────────────────────────────────────────────────────

def _mock_emotion(text: str):
    emotions = ["anxiety", "depression", "stress", "burnout", "relationship", "grief", "neutral"]
    np.random.seed(len(text))
    probs = np.random.dirichlet(np.ones(len(emotions)) * 2)
    emotion_probs = dict(zip(emotions, probs.tolist()))
    primary = max(emotion_probs, key=emotion_probs.get)
    return primary, float(emotion_probs[primary]), emotion_probs


def _mock_suicidal(text: str):
    keywords = ["end my life", "kill myself", "don't want to live", "suicide"]
    hit = any(kw in text.lower() for kw in keywords)
    conf = 0.91 if hit else 0.05
    return {"label": "suicidal" if hit else "non-suicidal", "confidence": conf, "is_suicidal": hit}


def _mock_problem(text: str):
    problems = ["anxiety", "depression", "stress", "burnout", "relationship", "grief", "neutral"]
    np.random.seed(len(text) + 1)
    probs = np.random.dirichlet(np.ones(len(problems)) * 2)
    return dict(zip(problems, probs.tolist()))


# ── Real API calls ───────────────────────────────────────────────────────────

def _call_predict(model: str, text: str) -> dict:
    payload = {"model": model, "text": text}
    r = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=10)
    r.raise_for_status()
    return r.json().get("result", {})


# ── Main entry ───────────────────────────────────────────────────────────────

def analyze_emotion(text: str):
    """
    Run all three classifiers against the backend.
    Returns (primary_emotion, confidence, emotion_probs, is_suicidal).
    Falls back to mock on any error.
    """
    try:
        # Emotion
        emotion_result = _call_predict("emotion", text)
        primary = max(emotion_result, key=emotion_result.get)
        confidence = float(emotion_result[primary])
        emotion_probs = {k: float(v) for k, v in emotion_result.items()}
    except Exception:
        primary, confidence, emotion_probs = _mock_emotion(text)

    try:
        suicidal_result = _call_predict("suicidal", text)
    except Exception:
        suicidal_result = _mock_suicidal(text)

    try:
        problem_result = _call_predict("problem", text)
    except Exception:
        problem_result = _mock_problem(text)

    is_suicidal = suicidal_result.get("is_suicidal", False)

    # Persist to session state
    st.session_state.emotion_probs   = emotion_probs
    st.session_state.suicidal_result = suicidal_result
    st.session_state.problem_result  = problem_result
    st.session_state.analysis_result = {
        "primary_emotion": primary,
        "confidence": confidence,
        "timestamp": datetime.now(),
    }
    st.session_state.history.append({
        "date": datetime.now(),
        "emotion": primary,
        "confidence": confidence,
    })

    return primary, confidence, emotion_probs, is_suicidal