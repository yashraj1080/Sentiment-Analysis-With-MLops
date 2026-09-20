"""
Model training and evaluation module for Sentiment Analysis.
Builds a Scikit-Learn TF-IDF + Logistic Regression pipeline, evaluates metrics,
and serializes artifacts for production API serving.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src.data.dataset import clean_text, get_train_test_split, load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "sentiment_pipeline.joblib"
DEFAULT_METADATA_PATH = MODELS_DIR / "model_metadata.json"


def build_sentiment_pipeline(
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
    c_param: float = 1.0,
    random_state: int = 42,
) -> Pipeline:
    """Construct an end-to-end TF-IDF + Logistic Regression Pipeline."""
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
        lowercase=True,
    )
    classifier = LogisticRegression(
        C=c_param,
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state,
        solver="lbfgs",
    )
    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", classifier),
    ])
    return pipeline


def evaluate_model(
    pipeline: Pipeline,
    X_test: List[str],
    y_test: List[str],
) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics."""
    y_pred = pipeline.predict(X_test)
    
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    recall = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
        "classification_report": report,
    }
    return metrics


def train_and_save_model(
    data_path: str | Path | None = None,
    output_model_path: str | Path | None = None,
    output_metadata_path: str | Path | None = None,
    model_version: str = "1.0.0",
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Load dataset, train NLP pipeline, evaluate performance, and save artifacts."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = Path(output_model_path) if output_model_path else DEFAULT_MODEL_PATH
    metadata_path = Path(output_metadata_path) if output_metadata_path else DEFAULT_METADATA_PATH

    logger.info("Loading dataset...")
    df = load_dataset(data_path)
    logger.info(f"Loaded dataset with {len(df)} total samples.")

    X_train, X_test, y_train, y_test = get_train_test_split(df)
    logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    logger.info("Building and fitting sentiment pipeline...")
    pipeline = build_sentiment_pipeline()
    pipeline.fit(X_train, y_train)

    logger.info("Evaluating model on test split...")
    metrics = evaluate_model(pipeline, X_test, y_test)
    logger.info(
        f"Evaluation Results -> Accuracy: {metrics['accuracy']:.4f} | "
        f"F1 Score: {metrics['f1_score']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}"
    )

    # Save trained pipeline artifact
    joblib.dump(pipeline, model_path)
    logger.info(f"Saved model pipeline to: {model_path}")

    # Build metadata
    vectorizer: TfidfVectorizer = pipeline.named_steps["tfidf"]
    classifier: LogisticRegression = pipeline.named_steps["clf"]

    metadata = {
        "model_version": model_version,
        "algorithm": "TF-IDF + Logistic Regression",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "classes": list(classifier.classes_),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "hyperparameters": {
            "ngram_range": vectorizer.ngram_range,
            "max_features": vectorizer.max_features,
            "sublinear_tf": vectorizer.sublinear_tf,
            "C": classifier.C,
            "max_iter": classifier.max_iter,
            "solver": classifier.solver,
        },
        "metrics": metrics,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved model metadata to: {metadata_path}")

    return pipeline, metadata


def load_model_pipeline(model_path: str | Path | None = None) -> Pipeline:
    """Load a serialized model pipeline from disk."""
    target_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not target_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {target_path}. Please train the model first by running `python -m src.model.train`."
        )
    return joblib.load(target_path)


def predict_sentiment(pipeline: Pipeline, raw_text: str) -> Dict[str, Any]:
    """Run inference on raw input text and return class, confidence, and probabilities."""
    cleaned = clean_text(raw_text)
    probs = pipeline.predict_proba([cleaned])[0]
    classes = pipeline.classes_
    best_idx = int(np.argmax(probs))
    
    sentiment_label = str(classes[best_idx])
    confidence = float(probs[best_idx])
    
    probabilities_dict = {
        str(cls_name): round(float(prob), 4)
        for cls_name, prob in zip(classes, probs)
    }
    
    return {
        "sentiment": sentiment_label,
        "confidence": round(confidence, 4),
        "probabilities": probabilities_dict,
    }


if __name__ == "__main__":
    train_and_save_model()
