"""
Unit tests for model pipeline construction, training, evaluation, and inference.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from sklearn.pipeline import Pipeline

from src.data.dataset import get_seed_dataset, get_train_test_split
from src.model.train import (
    build_sentiment_pipeline,
    evaluate_model,
    load_model_pipeline,
    predict_sentiment,
    train_and_save_model,
)


def test_build_sentiment_pipeline():
    pipeline = build_sentiment_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert "tfidf" in pipeline.named_steps
    assert "clf" in pipeline.named_steps


def test_train_and_save_model(tmp_path: Path):
    tmp_model_path = tmp_path / "test_model.joblib"
    tmp_meta_path = tmp_path / "test_meta.json"

    pipeline, metadata = train_and_save_model(
        output_model_path=tmp_model_path,
        output_metadata_path=tmp_meta_path,
        model_version="test-1.0",
    )

    assert tmp_model_path.exists()
    assert tmp_meta_path.exists()
    assert metadata["model_version"] == "test-1.0"
    assert metadata["metrics"]["accuracy"] >= 0.85
    assert metadata["metrics"]["f1_score"] >= 0.85

    # Test loading from disk
    loaded_pipeline = load_model_pipeline(tmp_model_path)
    assert isinstance(loaded_pipeline, Pipeline)


def test_predict_sentiment_positive_and_negative():
    pipeline = build_sentiment_pipeline()
    df = get_seed_dataset()
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    pipeline.fit(X_train, y_train)

    # Clearly positive text
    pos_res = predict_sentiment(pipeline, "This is an extraordinary masterpiece! I loved it so much.")
    assert pos_res["sentiment"] == "positive"
    assert pos_res["confidence"] > 0.5
    assert pos_res["probabilities"]["positive"] > pos_res["probabilities"]["negative"]

    # Clearly negative text
    neg_res = predict_sentiment(pipeline, "Worst disaster ever. Horrible acting, terrible waste of money.")
    assert neg_res["sentiment"] == "negative"
    assert neg_res["confidence"] > 0.5
    assert neg_res["probabilities"]["negative"] > neg_res["probabilities"]["positive"]


def test_model_missing_file_raises_error(tmp_path: Path):
    non_existent = tmp_path / "does_not_exist.joblib"
    with pytest.raises(FileNotFoundError):
        load_model_pipeline(non_existent)
