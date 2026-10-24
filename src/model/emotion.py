"""Multi-label emotion recognition trained on the GoEmotions corpus."""

from __future__ import annotations

import csv
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.dataset import clean_text

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "raw" / "goemotions"
MODELS_DIR = BASE_DIR / "models"
EMOTION_MODEL_PATH = MODELS_DIR / "emotion_pipeline.joblib"
EMOTION_METADATA_PATH = MODELS_DIR / "emotion_metadata.json"


def read_goemotions_split(path: str | Path, labels: list[str]) -> tuple[list[str], np.ndarray]:
    """Read official TSV rows into text and a 27-column multi-label matrix."""
    texts: list[str] = []
    targets: list[list[int]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as source:
        for row in csv.reader(source, delimiter="\t"):
            if len(row) < 2:
                continue
            text, label_ids = row[0], row[1]
            if not text.strip():
                continue
            target = [0] * len(labels)
            if label_ids.strip():
                for label_id in label_ids.split(","):
                    index = int(label_id)
                    if 0 <= index < len(labels):
                        target[index] = 1
            texts.append(clean_text(text))
            targets.append(target)
    if not texts:
        raise ValueError(f"No usable GoEmotions rows found in {path}")
    return texts, np.asarray(targets, dtype=np.uint8)


def build_emotion_pipeline(max_features: int = 30000) -> Pipeline:
    """Build a sparse TF-IDF, one-vs-rest logistic model for overlapping emotions."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_features,
            min_df=2,
            sublinear_tf=True,
            lowercase=True,
        )),
        ("clf", OneVsRestClassifier(LogisticRegression(
            C=2.0,
            class_weight="balanced",
            max_iter=500,
            solver="liblinear",
            random_state=42,
        ), n_jobs=1)),
    ])


def tune_thresholds(probabilities: np.ndarray, targets: np.ndarray) -> list[float]:
    """Select per-emotion thresholds on the dev split by F1, never on test data."""
    thresholds: list[float] = []
    candidates = np.arange(0.10, 0.81, 0.025)
    for index in range(targets.shape[1]):
        truth = targets[:, index]
        if truth.sum() == 0:
            thresholds.append(0.5)
            continue
        scores = [f1_score(truth, probabilities[:, index] >= threshold, zero_division=0) for threshold in candidates]
        thresholds.append(round(float(candidates[int(np.argmax(scores))]), 3))
    return thresholds


def evaluate_emotion_model(
    targets: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
    thresholds: list[float],
) -> dict[str, Any]:
    predictions = (probabilities >= np.asarray(thresholds)).astype(np.uint8)
    return {
        "macro_f1": round(float(f1_score(targets, predictions, average="macro", zero_division=0)), 4),
        "micro_f1": round(float(f1_score(targets, predictions, average="micro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(targets, predictions, average="weighted", zero_division=0)), 4),
        "micro_precision": round(float(precision_score(targets, predictions, average="micro", zero_division=0)), 4),
        "micro_recall": round(float(recall_score(targets, predictions, average="micro", zero_division=0)), 4),
        "hamming_loss": round(float(hamming_loss(targets, predictions)), 4),
        "subset_accuracy": round(float(accuracy_score(targets, predictions)), 4),
        "per_emotion": classification_report(
            targets, predictions, target_names=labels, output_dict=True, zero_division=0
        ),
    }


def train_and_save_emotion_model(
    data_dir: str | Path = DATA_DIR,
    output_model_path: str | Path = EMOTION_MODEL_PATH,
    output_metadata_path: str | Path = EMOTION_METADATA_PATH,
) -> tuple[Pipeline, dict[str, Any]]:
    """Train, tune on dev, evaluate once on test, and serialize the emotion model."""
    data_dir = Path(data_dir)
    labels_file = data_dir / "emotions.txt"
    if not labels_file.exists():
        raise FileNotFoundError("GoEmotions data is missing. Run `python scripts/download_goemotions.py` first.")
    labels = [line.strip() for line in labels_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(labels) != 28 or labels[-1] != "neutral":
        raise ValueError(f"Expected 27 emotions followed by neutral (28 labels), found {len(labels)} in {labels_file}")

    train_x, train_y = read_goemotions_split(data_dir / "train.tsv", labels)
    dev_x, dev_y = read_goemotions_split(data_dir / "dev.tsv", labels)
    test_x, test_y = read_goemotions_split(data_dir / "test.tsv", labels)
    pipeline = build_emotion_pipeline()
    logger.info("Training multi-label emotion model on %s comments and %s labels", len(train_x), len(labels))
    pipeline.fit(train_x, train_y)

    dev_probabilities = pipeline.predict_proba(dev_x)
    thresholds = tune_thresholds(dev_probabilities, dev_y)
    test_probabilities = pipeline.predict_proba(test_x)
    metrics = evaluate_emotion_model(test_y, test_probabilities, labels, thresholds)

    model_path = Path(output_model_path)
    metadata_path = Path(output_metadata_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metadata = {
        "model_version": "1.0.0",
        "task": "multi-label emotion classification",
        "dataset": "Google Research GoEmotions agreement-filtered official split",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_samples": len(train_x),
        "dev_samples": len(dev_x),
        "test_samples": len(test_x),
        "labels": labels,
        "thresholds_tuned_on": "dev",
        "thresholds": dict(zip(labels, thresholds)),
        "algorithm": "TF-IDF word and bigram features + one-vs-rest balanced Logistic Regression",
        "metrics": metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Test macro-F1: %.4f | micro-F1: %.4f", metrics["macro_f1"], metrics["micro_f1"])
    return pipeline, metadata


def load_emotion_model(model_path: str | Path = EMOTION_MODEL_PATH) -> Pipeline:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Emotion model not found at {path}; train it with `python -m src.model.emotion`.")
    return joblib.load(path)


def predict_emotions(
    pipeline: Pipeline,
    raw_text: str,
    labels: list[str],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    clauses = [part.strip(" ,;:-") for part in re.split(
        r"(?i)\b(?:but|however|although|though|yet)\b|(?<=[.!?;])\s+|\n+", raw_text
    ) if len(part.split()) >= 2]
    clauses = clauses[:6]
    inputs = [clean_text(raw_text), *(clean_text(clause) for clause in clauses)]
    all_probabilities = pipeline.predict_proba(inputs)

    def rank_emotions(probabilities: np.ndarray) -> list[dict[str, Any]]:
        ranked = sorted(
            ({"emotion": label, "confidence": round(float(score), 4)} for label, score in zip(labels, probabilities)),
            key=lambda item: item["confidence"],
            reverse=True,
        )
        if any(
            item["emotion"] != "neutral"
            and item["confidence"] >= thresholds.get(item["emotion"], 0.5)
            for item in ranked
        ):
            ranked = [item for item in ranked if item["emotion"] != "neutral"]
        return ranked

    def select_emotions(ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
        selected = [item for item in ranked if item["confidence"] >= thresholds.get(item["emotion"], 0.5)]
        selected_emotions = [item for item in selected if item["emotion"] != "neutral"]
        # Neutral is mutually exclusive in GoEmotions; do not show it alongside
        # non-neutral predictions from the independent one-vs-rest classifiers.
        return selected_emotions if selected_emotions else selected

    ranked = rank_emotions(all_probabilities[0])
    selected = select_emotions(ranked)
    segments = []
    for clause, probabilities in zip(clauses, all_probabilities[1:]):
        clause_ranked = rank_emotions(probabilities)
        segments.append({
            "text": clause,
            "emotions": clause_ranked[:3],
        })
    return {
        "emotions": selected,
        "top_emotions": ranked[:3],
        "segments": segments,
        "is_neutral": bool(selected and selected[0]["emotion"] == "neutral"),
        "note": "Emotion scores are independent; more than one emotion may be present.",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    train_and_save_emotion_model()
