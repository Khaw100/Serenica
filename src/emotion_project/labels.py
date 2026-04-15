GOEMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]

label2id = {l: i for i, l in enumerate(GOEMOTIONS)}
id2label = {i: l for i, l in enumerate(GOEMOTIONS)}
NUM_LABELS = len(GOEMOTIONS)
