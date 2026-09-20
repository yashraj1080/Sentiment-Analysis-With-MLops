"""
Download and extract the full IMDB Movie Reviews Dataset (50,000 reviews).
This script provides the dataset mentioned in Reference [8] of the project report:
"Learning Word Vectors for Sentiment Analysis" (Maas et al., ACL 2011).
"""

from __future__ import annotations

import os
import tarfile
import urllib.request
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
IMDB_TAR_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
OUTPUT_CSV_PATH = DATA_RAW_DIR / "imdb_sentiment_50k.csv"


def download_and_prepare_imdb(limit_per_class: int | None = None) -> Path:
    """Download IMDB tar.gz archive, extract positive and negative reviews,
    and convert to a structured CSV format (text, sentiment).
    """
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    if OUTPUT_CSV_PATH.exists():
        print(f"IMDB dataset already prepared at: {OUTPUT_CSV_PATH}")
        return OUTPUT_CSV_PATH

    tar_path = DATA_RAW_DIR / "aclImdb_v1.tar.gz"
    if not tar_path.exists():
        print(f"Downloading IMDB dataset from {IMDB_TAR_URL}...")
        urllib.request.urlretrieve(IMDB_TAR_URL, tar_path)
        print("Download complete.")

    records = []
    print("Extracting and parsing reviews...")
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile() or not member.name.endswith(".txt"):
                continue
            
            # Paths look like aclImdb/train/pos/..., aclImdb/test/neg/...
            parts = member.name.split("/")
            if len(parts) >= 4 and parts[2] in ("pos", "neg"):
                sentiment = "positive" if parts[2] == "pos" else "negative"
                f = tar.extractfile(member)
                if f:
                    content = f.read().decode("utf-8", errors="replace").strip()
                    records.append({"text": content, "sentiment": sentiment})

    df = pd.DataFrame(records)
    print(f"Extracted {len(df)} total reviews. Class distribution:")
    print(df["sentiment"].value_counts())

    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Saved full IMDB dataset to: {OUTPUT_CSV_PATH}")
    return OUTPUT_CSV_PATH


if __name__ == "__main__":
    download_and_prepare_imdb()
