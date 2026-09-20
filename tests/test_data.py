"""
Unit tests for data preprocessing, text normalization, and dataset splitting.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.dataset import clean_text, get_seed_dataset, get_train_test_split, load_dataset


def test_clean_text_html_removal():
    raw_html = "<p>This is a <b>fantastic</b> movie!<br /></p>"
    cleaned = clean_text(raw_html)
    assert "<" not in cleaned
    assert ">" not in cleaned
    assert "fantastic" in cleaned
    assert "movie" in cleaned


def test_clean_text_url_removal():
    raw_url = "Check out the review at https://example.com/movie-review! It was terrible."
    cleaned = clean_text(raw_url)
    assert "http" not in cleaned
    assert "example.com" not in cleaned
    assert "terrible" in cleaned


def test_clean_text_case_and_whitespace():
    raw = "   VERY   loud and    AMAZING!   "
    cleaned = clean_text(raw)
    assert cleaned == "very loud and amazing!"


def test_clean_text_empty_and_non_string():
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text(12345) == ""


def test_get_seed_dataset_balance():
    df = get_seed_dataset()
    assert isinstance(df, pd.DataFrame)
    assert "text" in df.columns
    assert "sentiment" in df.columns
    assert len(df) >= 100
    
    pos_count = (df["sentiment"] == "positive").sum()
    neg_count = (df["sentiment"] == "negative").sum()
    assert pos_count == neg_count, "Dataset should be balanced between positive and negative classes"


def test_get_train_test_split():
    df = get_seed_dataset()
    X_train, X_test, y_train, y_test = get_train_test_split(df, test_size=0.2, random_state=42)
    
    total = len(df)
    assert len(X_train) + len(X_test) == total
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    assert len(X_test) == pytest.approx(int(total * 0.2), abs=2)
    
    # Assert stratification
    train_pos = y_train.count("positive") / len(y_train)
    test_pos = y_test.count("positive") / len(y_test)
    assert train_pos == pytest.approx(test_pos, abs=0.05)
