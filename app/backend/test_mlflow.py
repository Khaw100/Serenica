"""Smoke-test script for MLflow models moved to `backend` folder."""

import argparse
import logging
import json
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.mlflow_loader import load_classifiers


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tracking-uri", type=str, default=None, help="MLflow tracking URI (override env)")
    p.add_argument("--stage", type=str, default="latest", help="Model stage: latest/Production/Staging")
    p.add_argument("--models", type=str, default="emotion,suicidal,problem", help="Comma-separated list of models to test")
    p.add_argument("--text", type=str, default=None, help="Text to classify (optional). If omitted, several sample texts will be used")
    return p.parse_args()


SAMPLES = [
    "I feel so happy and excited today!",
    "I've been feeling anxious and worried for days.",
    "I want to end my life and nobody understands me.",
]


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    texts: List[str] = [args.text] if args.text else SAMPLES

    try:
        clfs = load_classifiers(tracking_uri=args.tracking_uri, stage=args.stage)
    except Exception as e:
        print(f"Failed to load classifiers from MLflow: {e}")
        raise

    for name in models:
        clf = clfs.get(name)
        if clf is None:
            print(f"Model '{name}' not found in loaded classifiers. Available: {list(clfs.keys())}")
            continue

        print("\n" + "=" * 40)
        print(f"Model: {name}")
        for t in texts:
            try:
                out = clf.predict(t)
            except TypeError:
                out = clf.predict(t)
            print(f"\nInput: {t}")
            print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
