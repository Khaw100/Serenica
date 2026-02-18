"""
main.py — Test semua classifier yang sudah ada

Jalankan dari root folder project:
    python main.py

Struktur folder:
    mental-health-ai/
    ├── models/
    │   ├── emotion_classifier/
    │   │   ├── config.json
    │   │   ├── model.safetensors
    │   │   ├── tokenizer.json
    │   │   ├── tokenizer_config.json
    │   │   └── emotion_config.pt
    │   └── suicidal_classifier/
    │       └── suicide_detection_pipeline.pkl
    ├── ml/
    │   ├── __init__.py
    │   ├── base_classifier.py
    │   ├── emotion_classifier.py
    │   └── suicidal_classifier.py
    └── main.py
"""

import logging
from ml.emotion_classifier  import EmotionClassifier
from ml.suicidal_classifier import SuicidalClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("Loading models...")
print("="*60)

emotion_clf = EmotionClassifier(
    model_dir="./models/emotion_classifier",
)

suicidal_clf = SuicidalClassifier(
    model_path="./models/suicidal_classifier/suicide_detection_pipeline.pkl",
    threshold=0.8,
)

print("\n✅ Semua model loaded!")
print(f"\n{emotion_clf}")
print(f"\n{suicidal_clf}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST CASES
# ─────────────────────────────────────────────────────────────────────────────

test_texts = [
    {
        "label": "Happy / Positive",
        "text" : "I feel so happy and grateful today! Life is beautiful.",
    },
    {
        "label": "Sad / Depressed (non-suicidal)",
        "text" : (
            "Lately I've been feeling completely empty and unmotivated. "
            "I go to work, come home, and scroll through my phone. "
            "I don't enjoy my hobbies anymore."
        ),
    },
    {
        "label": "Suicidal ideation",
        "text" : (
            "I don't even know where to start. Every day feels heavier. "
            "I keep thinking about how peaceful it would be to just disappear. "
            "I've been searching online for ways to end my life. "
            "I don't think anyone would notice if I was gone."
        ),
    },
    {
        "label": "Recovery / Hopeful",
        "text" : (
            "Last year was the darkest period of my life. "
            "But I started therapy and it's helping. "
            "I went for a walk this morning and felt a little peace. "
            "I'm starting to believe things can get better."
        ),
    },
    {
        "label": "Anxious / Nervous",
        "text" : (
            "I keep replaying conversations in my head, wondering if I said "
            "the wrong things. Small mistakes feel huge and I end up blaming "
            "myself over and over. It makes me anxious about the future."
        ),
    },
]

print("\n" + "="*60)
print("RESULTS")
print("="*60)

for tc in test_texts:
    text  = tc["text"]
    label = tc["label"]

    emotion_result  = emotion_clf.predict(text)
    suicidal_result = suicidal_clf.predict(text)

    suicidal_flag = "⚠️  SUICIDAL" if suicidal_result["is_suicidal"] else "✅ NON-SUICIDAL"

    print(f"\n┌─ [{label}]")
    print(f"│  Text     : {text[:70]}...")
    print(f"│  Emotion  : {emotion_result}")
    print(f"│  Suicidal : {suicidal_flag} (conf={suicidal_result['confidence']})")
    print(f"└{'─'*58}")

# ─────────────────────────────────────────────────────────────────────────────
# BATCH TEST
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("BATCH PREDICT")
print("="*60)

batch_texts = [
    "I want to end my life",
    "I feel sad but I'm working through it with my therapist",
    "Everything is pointless, I see no reason to continue",
    "I'm nervous about my exam tomorrow",
    "I love spending time with my family, it makes me happy",
]

emotion_batch  = emotion_clf.predict_batch(batch_texts)
suicidal_batch = suicidal_clf.predict_batch(batch_texts)

for text, em, su in zip(batch_texts, emotion_batch, suicidal_batch):
    flag = "⚠️ " if su["is_suicidal"] else "✅ "
    print(f"\n  Text     : '{text}'")
    print(f"  Emotion  : {em}")
    print(f"  {flag}Suicidal: {su['label']} ({su['confidence']})")