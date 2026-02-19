"""
push_to_mlflow.py

Push model yang sudah ada di local ke MLflow Model Registry.
Evaluasi metrics bersifat opsional — bisa diisi manual lewat CLI.

Contoh penggunaan:
    # Push semua, tanpa metrics
    python push_to_mlflow.py

    # Push problem classifier + input metrics manual
    python push_to_mlflow.py --model problem --accuracy 0.8923 --f1 0.8791

    # Push emotion classifier + metrics
    python push_to_mlflow.py --model emotion --accuracy 0.91 --f1 0.89

    # Push semua + metrics (berlaku untuk semua model yang di-push)
    python push_to_mlflow.py --model all --accuracy 0.88 --f1 0.86
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import mlflow
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME     = "mental-health-ai"


# ═════════════════════════════════════════════════════════════════════════════
# PythonModel Wrappers
# ═════════════════════════════════════════════════════════════════════════════

class HuggingFaceClassifierWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        model_path     = context.artifacts["hf_model_dir"]
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device).eval()

        cfg = self.model.config
        if hasattr(cfg, "id2label") and len(cfg.id2label) > 1:
            self.label_names = [cfg.id2label[i] for i in range(len(cfg.id2label))]
        else:
            for fname in ["model_config.pt", "emotion_config.pt", "problem_config.pt"]:
                pt = Path(model_path) / fname
                if pt.exists():
                    saved = torch.load(str(pt), map_location="cpu")
                    self.label_names = saved.get("label_names", [])
                    break

    def predict(self, context, model_input):
        import pandas as pd
        texts = model_input["text"].tolist()
        enc   = self.tokenizer(texts, return_tensors="pt",
                               truncation=True, padding=True, max_length=128)
        enc   = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return pd.DataFrame(probs, columns=self.label_names)


class SklearnPipelineWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        import joblib
        self.pipeline  = joblib.load(context.artifacts["pkl_path"])
        self.threshold = 0.8  # threshold disimpan di params, bukan artifacts

    def predict(self, context, model_input):
        import pandas as pd
        texts  = model_input["text"].tolist()
        probs  = self.pipeline.predict_proba(texts)[:, 1]
        labels = ["suicidal" if p >= self.threshold else "non-suicidal" for p in probs]
        return pd.DataFrame({
            "label"      : labels,
            "confidence" : [round(float(p), 4) for p in probs],
            "is_suicidal": [p >= self.threshold for p in probs],
        })


# ═════════════════════════════════════════════════════════════════════════════
# Helper
# ═════════════════════════════════════════════════════════════════════════════

def _read_hf_config(model_dir: Path) -> dict:
    """Baca metadata dari config.json (tidak wajib ada)."""
    info = {}
    config_json = model_dir / "config.json"
    if config_json.exists():
        with open(config_json) as f:
            hf_cfg = json.load(f)
        info["base_model"] = hf_cfg.get("_name_or_path", "unknown")
        info["num_labels"] = hf_cfg.get("num_labels", "unknown")

    for fname in ["model_config.pt", "emotion_config.pt", "problem_config.pt"]:
        pt = model_dir / fname
        if pt.exists():
            cfg = torch.load(str(pt), map_location="cpu")
            info["label_names"] = cfg.get("label_names", [])
            info["model_type"]  = cfg.get("model_type", "roberta")
            # Ambil dari .pt kalau ada, tapi tetap bisa di-override manual
            info["accuracy_from_pt"] = cfg.get("test_accuracy", None)
            info["f1_from_pt"]       = cfg.get("test_macro_f1", None)
            break

    return info


def _log_metrics_if_any(
    accuracy: Optional[float],
    f1: Optional[float],
    meta: dict,
):
    """
    Log metrics ke MLflow.
    Prioritas: nilai manual (dari CLI) > nilai dari .pt config > skip.
    """
    final_acc = accuracy if accuracy is not None else meta.get("accuracy_from_pt")
    final_f1  = f1       if f1       is not None else meta.get("f1_from_pt")

    if final_acc is not None:
        mlflow.log_metric("test_accuracy", final_acc)
        logger.info(f"   test_accuracy = {final_acc}")
    else:
        logger.info("   test_accuracy = (tidak diisi, dilewat)")

    if final_f1 is not None:
        mlflow.log_metric("test_macro_f1", final_f1)
        logger.info(f"   test_macro_f1 = {final_f1}")
    else:
        logger.info("   test_macro_f1 = (tidak diisi, dilewat)")


# ═════════════════════════════════════════════════════════════════════════════
# Push functions
# ═════════════════════════════════════════════════════════════════════════════

def push_emotion_classifier(
    accuracy: Optional[float] = None,
    f1: Optional[float] = None,
):
    model_dir       = Path("./models/emotion_classifier")
    registered_name = "emotion-classifier"

    logger.info(f"📦 Pushing EmotionClassifier dari {model_dir}...")
    meta = _read_hf_config(model_dir)

    with mlflow.start_run(run_name="push_emotion_classifier"):
        mlflow.log_param("model_type",  meta.get("model_type", "roberta"))
        mlflow.log_param("base_model",  meta.get("base_model", "unknown"))
        mlflow.log_param("num_labels",  meta.get("num_labels", "unknown"))
        mlflow.log_param("label_names", str(meta.get("label_names", [])))

        _log_metrics_if_any(accuracy, f1, meta)

        mlflow.pyfunc.log_model(
            artifact_path         = "model",
            python_model          = HuggingFaceClassifierWrapper(),
            artifacts             = {"hf_model_dir": str(model_dir.resolve())},
            registered_model_name = registered_name,
            pip_requirements      = ["torch", "transformers", "pandas"],
        )

        run_id = mlflow.active_run().info.run_id
        logger.info(f"✅ EmotionClassifier pushed! run_id={run_id}")
        return run_id


def push_suicidal_classifier(
    accuracy: Optional[float] = None,
    f1: Optional[float] = None,
):
    pkl_path        = Path("./models/suicidal_classifier/suicide_detection_pipeline.pkl")
    registered_name = "suicidal-classifier"

    logger.info(f"📦 Pushing SuicidalClassifier dari {pkl_path}...")

    with mlflow.start_run(run_name="push_suicidal_classifier"):
        mlflow.log_param("model_type", "sklearn_pipeline")
        mlflow.log_param("threshold",  0.8)

        _log_metrics_if_any(accuracy, f1, meta={})

        mlflow.pyfunc.log_model(
            artifact_path         = "model",
            python_model          = SklearnPipelineWrapper(),
            artifacts             = {"pkl_path": str(pkl_path.resolve())},  # threshold bukan file, jangan masuk sini
            registered_model_name = registered_name,
            pip_requirements      = ["scikit-learn", "joblib", "pandas"],
        )

        run_id = mlflow.active_run().info.run_id
        logger.info(f"✅ SuicidalClassifier pushed! run_id={run_id}")
        return run_id


def push_problem_classifier(
    accuracy: Optional[float] = None,
    f1: Optional[float] = None,
):
    model_dir       = Path("./models/problem_classifier")
    registered_name = "problem-classifier"

    logger.info(f"📦 Pushing ProblemClassifier dari {model_dir}...")
    meta = _read_hf_config(model_dir)

    with mlflow.start_run(run_name="push_problem_classifier"):
        mlflow.log_param("model_type",  meta.get("model_type", "roberta"))
        mlflow.log_param("base_model",  meta.get("base_model", "unknown"))
        mlflow.log_param("num_labels",  meta.get("num_labels", "unknown"))
        mlflow.log_param("label_names", str(meta.get("label_names", [])))

        _log_metrics_if_any(accuracy, f1, meta)

        mlflow.pyfunc.log_model(
            artifact_path         = "model",
            python_model          = HuggingFaceClassifierWrapper(),
            artifacts             = {"hf_model_dir": str(model_dir.resolve())},
            registered_model_name = registered_name,
            pip_requirements      = ["torch", "transformers", "pandas"],
        )

        run_id = mlflow.active_run().info.run_id
        logger.info(f"✅ ProblemClassifier pushed! run_id={run_id}")
        return run_id


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Push local models ke MLflow Registry",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Contoh:
  python push_to_mlflow.py                                         # push semua, no metrics
  python push_to_mlflow.py --model problem --accuracy 0.89 --f1 0.87
  python push_to_mlflow.py --model emotion --accuracy 0.91 --f1 0.89
  python push_to_mlflow.py --model suicidal --accuracy 0.95
        """,
    )
    parser.add_argument(
        "--model",
        choices=["emotion", "suicidal", "problem", "all"],
        default="all",
    )
    parser.add_argument(
        "--accuracy",
        type=float,
        default=None,
        help="Test accuracy (0.0 - 1.0). Opsional.",
    )
    parser.add_argument(
        "--f1",
        type=float,
        default=None,
        help="Test macro F1 score (0.0 - 1.0). Opsional.",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info(f"🚀 MLflow URI  : {MLFLOW_TRACKING_URI}")
    logger.info(f"   Experiment  : {EXPERIMENT_NAME}")
    logger.info(f"   Model       : {args.model}")
    logger.info(f"   Accuracy    : {args.accuracy or '(tidak diisi)'}")
    logger.info(f"   F1 Score    : {args.f1 or '(tidak diisi)'}")
    print()

    results = {}

    # if args.model in ("emotion", "all"):
    #     results["emotion"]  = push_emotion_classifier(args.accuracy, args.f1)

    # if args.model in ("suicidal", "all"):
    #     results["suicidal"] = push_suicidal_classifier(args.accuracy, args.f1)

    if args.model in ("problem", "all"):
        results["problem"]  = push_problem_classifier(args.accuracy, args.f1)

    print()
    print("=" * 55)
    print("✅ PUSH SELESAI")
    print("=" * 55)
    for name, run_id in results.items():
        print(f"  {name:<10} → run_id: {run_id}")
    print()
    print(f"MLflow UI → {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()