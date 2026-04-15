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
    │   ├── suicidal_classifier/
    │   │   └── suicide_detection_pipeline.pkl
    │   └── problem_classifier/
    │       ├── config.json
    │       ├── model.safetensors
    │       ├── tokenizer.json
    │       ├── tokenizer_config.json
    │       └── model_config.pt
    ├── ml/
    │   ├── __init__.py
    │   ├── base_classifier.py
    │   ├── emotion_classifier.py
    │   ├── suicidal_classifier.py
    │   └── problem_classifier.py
    └── main.py
"""

import logging
from ml.emotion_classifier  import EmotionClassifier
from ml.suicidal_classifier import SuicidalClassifier
from ml.problem_classifier  import ProblemClassifier

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

problem_clf = ProblemClassifier(
    model_dir="./models/problem_classifier",
)

print("\n✅ Semua model loaded!")
print(f"\n{emotion_clf}")
print(f"\n{suicidal_clf}")
print(f"\n{problem_clf}")

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
    problem_result  = problem_clf.predict(text)

    suicidal_flag = "⚠️  SUICIDAL" if suicidal_result["is_suicidal"] else "✅ NON-SUICIDAL"

    print(f"\n┌─ [{label}]")
    print(f"│  Text     : {text[:70]}...")
    print(f"│  Emotion  : {emotion_result}")
    print(f"│  Suicidal : {suicidal_flag} (conf={suicidal_result['confidence']})")
    print(f"│  Problem  : {problem_result}")
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
problem_batch  = problem_clf.predict_batch(batch_texts)

for text, em, su, pr in zip(batch_texts, emotion_batch, suicidal_batch, problem_batch):
    flag = "⚠️ " if su["is_suicidal"] else "✅ "
    print(f"\n  Text     : '{text}'")
    print(f"  Emotion  : {em}")
    print(f"  {flag}Suicidal: {su['label']} ({su['confidence']})")
    print(f"  Problem  : {pr}")

# ─────────────────────────────────────────────────────────────────────────────
# PROBLEM CLASSIFIER — DEDICATED TEST
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("PROBLEM CLASSIFIER — DEDICATED TEST")
print("="*60)

problem_test_texts = [
    {
        "label": "Anxiety",
        "text" : "I can't stop worrying, my heart races all the time and I keep overthinking everything.",
    },
    {
        "label": "Burnout",
        "text" : (
            "I used to love my job but now I feel completely drained. "
            "I have nothing left to give. Every morning I dread going to work."
        ),
    },
    {
        "label": "Depression",
        "text" : "I've lost all motivation, nothing feels meaningful anymore. I just feel empty inside.",
    },
    {
        "label": "Grief",
        "text" : "I lost my best friend last month and I can't accept they're gone. I miss them every day.",
    },
    {
        "label": "Relationship Issues",
        "text" : (
            "My relationship is falling apart and I don't know what to do. "
            "We argue constantly and I feel so disconnected from my partner."
        ),
    },
    {
        "label": "Work Stress",
        "text" : "Deadlines are piling up, I feel completely overwhelmed at work and can't keep up.",
    },
    {
        "label": "None / Neutral",
        "text" : "Today was a pretty normal day, had coffee and went for a walk.",
    },
    {
        "label": "Mixed (multi-issue)",
        "text" : (
            "Lately my relationship has been stressing me out more than anything else. "
            "We've been arguing about small things and it never really gets resolved because we're both busy and tired. "
            "I'm juggling work deadlines during the day and at night I keep thinking about the last conversation we had. "
            "It feels like I'm failing both at my job and in my relationship at the same time."
        ),
    },
]

for tc in problem_test_texts:
    text  = tc["text"]
    label = tc["label"]

    top1   = problem_clf.predict(text)
    top3   = problem_clf.get_top_k(text, k=3)

    print(f"\n┌─ [{label}]")
    print(f"│  Text : {text[:70]}...")
    print(f"│  Top-1: {top1}")
    print(f"│  Top-3: {top3}")
    print(f"└{'─'*58}")