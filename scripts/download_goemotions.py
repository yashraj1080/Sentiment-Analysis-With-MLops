"""Download the official GoEmotions agreement-filtered dataset splits."""

from __future__ import annotations

import urllib.request
from pathlib import Path


BASE_URL = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/"
FILES = ("train.tsv", "dev.tsv", "test.tsv", "emotions.txt")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "goemotions"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        destination = OUTPUT_DIR / filename
        if destination.exists() and destination.stat().st_size:
            print(f"Already downloaded: {destination}")
            continue
        print(f"Downloading {filename}...")
        request = urllib.request.Request(
            BASE_URL + filename,
            headers={"User-Agent": "sentiment-mlops/1.0"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            destination.write_bytes(response.read())
        print(f"Saved {destination}")


if __name__ == "__main__":
    main()
