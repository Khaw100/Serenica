PSYCHOLOGISTS = [
    {
        "name": "Dr. Sarah Chen",
        "spec": "Anxiety & Stress Management",
        "bio":  "Specializes in CBT for anxiety disorders and chronic stress.",
        "expertise": {
            # emotion model weights
            "nervousness": 0.9, "fear": 0.8, "worry": 0.7,
            # problem model weights
            "anxiety": 1.0, "stress": 0.8,
        },
    },
    {
        "name": "Dr. Michael Rodriguez",
        "spec": "Cognitive Behavioral Therapy",
        "bio":  "Expert in restructuring negative thought patterns across mood disorders.",
        "expertise": {
            "sadness": 0.7, "nervousness": 0.7, "disappointment": 0.6,
            "anxiety": 0.8, "depression": 0.8, "stress": 0.7,
        },
    },
    {
        "name": "Dr. Emily Johnson",
        "spec": "Depression & Mood Disorders",
        "bio":  "Focuses on mood regulation, grief processing, and long-term recovery.",
        "expertise": {
            "sadness": 1.0, "grief": 1.0, "remorse": 0.8, "disappointment": 0.7,
            "depression": 1.0, "grief": 1.0, "burnout": 0.6,
        },
    },
    {
        "name": "Dr. David Kim",
        "spec": "Relationship & Interpersonal Therapy",
        "bio":  "Helps navigate interpersonal conflicts, communication, and attachment issues.",
        "expertise": {
            "anger": 0.8, "annoyance": 0.7, "disapproval": 0.7, "remorse": 0.6,
            "relationship": 1.0,
        },
    },
    {
        "name": "Dr. Ayu Pertiwi",
        "spec": "Burnout & Occupational Stress",
        "bio":  "Focuses on work-life balance, burnout recovery, and resilience.",
        "expertise": {
            "nervousness": 0.6, "sadness": 0.5, "disappointment": 0.6,
            "burnout": 1.0, "stress": 0.9, "anxiety": 0.5,
        },
    },
    {
        "name": "Dr. James Tan",
        "spec": "Crisis Intervention & Trauma",
        "bio":  "Trained in crisis response, suicidal ideation, and trauma-informed care.",
        "expertise": {
            "grief": 0.8, "sadness": 0.7, "fear": 0.7, "remorse": 0.6,
            "depression": 0.7, "grief": 0.8,
            # special: selalu dapat boost kalau suicidal
            "_suicidal_boost": 1.0,
        },
    },
    {
        "name": "Dr. Rina Halim",
        "spec": "Mindfulness & Emotional Regulation",
        "bio":  "Uses mindfulness-based approaches for emotional balance and self-awareness.",
        "expertise": {
            "confusion": 0.7, "realization": 0.6, "neutral": 0.5,
            "stress": 0.6, "anxiety": 0.5, "neutral": 0.6,
        },
    },
    {
        "name": "Dr. Kevin Lim",
        "spec": "Grief & Loss Counseling",
        "bio":  "Specializes in bereavement, loss, and life transitions.",
        "expertise": {
            "grief": 1.0, "sadness": 0.8, "remorse": 0.7,
            "grief": 1.0, "depression": 0.5,
        },
    },
]

def compute_match_score(
    psychologist: dict,
    emotion_probs: dict,
    problem_probs: dict,
    is_suicidal: bool,
) -> float:
    expertise = psychologist["expertise"]
    score = 0.0

    # Kontribusi dari emotion model (bobot 0.4)
    for label, prob in emotion_probs.items():
        if label in expertise:
            score += prob * expertise[label] * 0.4

    # Kontribusi dari problem model (bobot 0.6 — lebih spesifik)
    for label, prob in problem_probs.items():
        if label in expertise:
            score += prob * expertise[label] * 0.6

    # Suicidal boost
    if is_suicidal and "_suicidal_boost" in expertise:
        score += expertise["_suicidal_boost"] * 0.5

    return round(min(score, 0.99), 2)


def get_matched_psychologists(
    emotion_probs: dict,
    problem_probs: dict,
    is_suicidal: bool,
    top_n: int = 4,
) -> list[dict]:
    """
    Return top_n psychologists sorted by match score.
    """
    results = []
    for psych in PSYCHOLOGISTS:
        score = compute_match_score(psych, emotion_probs, problem_probs, is_suicidal)
        results.append({**psych, "match": score})

    # Kalau suicidal, Dr. James Tan selalu masuk top hasil
    results.sort(key=lambda x: x["match"], reverse=True)

    # Pastikan crisis specialist muncul kalau is_suicidal
    if is_suicidal:
        crisis_doc = next((p for p in results if "_suicidal_boost" in p["expertise"]), None)
        top = results[:top_n]
        if crisis_doc and crisis_doc not in top:
            top[-1] = crisis_doc  # ganti yang paling bawah
        return top

    return results[:top_n]