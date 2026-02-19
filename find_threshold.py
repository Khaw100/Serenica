# ── Jalankan di Colab / lokal untuk cari threshold optimal ──────────────────
import joblib

pipeline = joblib.load("./models/suicidal_classifier/suicide_detection_pipeline.pkl")

# Test cases dengan expected label
test_cases = [
    # (text, expected)
    ("I feel so happy and grateful today! Life is beautiful.",                          "non-suicidal"),
    ("Lately I've been feeling empty and unmotivated. I don't enjoy my hobbies.",       "non-suicidal"),
    ("I started therapy and it's helping. I went for a walk and felt peace.",           "non-suicidal"),
    ("I keep replaying conversations. Small mistakes feel huge, I blame myself.",       "non-suicidal"),
    ("I feel sad but I'm working through it with my therapist",                         "non-suicidal"),
    ("I'm nervous about my exam tomorrow",                                              "non-suicidal"),
    ("I love spending time with my family, it makes me happy",                          "non-suicidal"),

    ("I keep thinking about how peaceful it would be to just disappear. "
     "I've been searching online for ways to end my life.",                             "suicidal"),
    ("I want to end my life",                                                           "suicidal"),
    ("Everything is pointless, I see no reason to continue",                            "suicidal"),
    ("I don't think anyone would notice if I was gone. I'm so tired of fighting.",      "suicidal"),
]

# Preprocess (sama seperti di SuicidalClassifier)
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords', quiet=True)

stop_words     = set(stopwords.words("english"))
protected      = {"i","me","my","myself","you","your","yours","he","him","his",
                  "she","her","hers","they","them","their"}
filtered_stops = stop_words - protected

def preprocess(text):
    words = text.split()
    text  = " ".join(w for w in words if w not in filtered_stops)
    text  = text.lower()
    text  = re.sub(r"https?://\S+|www\.\S+", "", text)
    text  = re.sub(r"\d+", "", text)
    text  = re.sub(r"[^\w\s]", "", text)
    text  = re.sub(r"\s+", " ", text).strip()
    return text

cleaned_texts = [preprocess(t) for t, _ in test_cases]
expected      = [e for _, e in test_cases]
probas        = pipeline.predict_proba(cleaned_texts)[:, 1]  # prob suicidal

# ── Sweep threshold ──────────────────────────────────────────────────────────
print(f"{'Threshold':>10} | {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4} | {'Precision':>10} {'Recall':>8} {'F1':>8}")
print("-" * 70)

best_f1  = 0
best_thr = 0.5

for thr in [i/100 for i in range(40, 95, 5)]:
    preds = ["suicidal" if p >= thr else "non-suicidal" for p in probas]

    tp = sum(p == "suicidal"     and e == "suicidal"     for p, e in zip(preds, expected))
    fp = sum(p == "suicidal"     and e == "non-suicidal" for p, e in zip(preds, expected))
    tn = sum(p == "non-suicidal" and e == "non-suicidal" for p, e in zip(preds, expected))
    fn = sum(p == "non-suicidal" and e == "suicidal"     for p, e in zip(preds, expected))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    marker = " ← best F1" if f1 > best_f1 else ""
    if f1 > best_f1:
        best_f1  = f1
        best_thr = thr

    print(f"{thr:>10.2f} | {tp:>4} {fp:>4} {tn:>4} {fn:>4} | {precision:>10.3f} {recall:>8.3f} {f1:>8.3f}{marker}")

print(f"\n✅ Best threshold : {best_thr}")
print(f"   Best F1        : {best_f1:.3f}")

# ── Detail per teks pada best threshold ──────────────────────────────────────
print(f"\n── Detail pada threshold={best_thr} ─────────────────────────────")
for (text, expected_label), prob in zip(test_cases, probas):
    pred  = "suicidal" if prob >= best_thr else "non-suicidal"
    match = "✅" if pred == expected_label else "❌"
    flag  = "⚠️ " if pred == "suicidal" else "  "
    print(f"{match} {flag} [{expected_label:>13}] conf={prob:.4f}  '{text[:55]}...'")