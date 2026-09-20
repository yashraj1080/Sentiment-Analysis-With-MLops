"""
One-click developer runner script.
Ensures model artifact is present (trains if missing), runs test suite,
and starts the local FastAPI development server with uvicorn.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "sentiment_pipeline.joblib"


def main():
    print("=" * 60)
    print(" Sentiment Analysis MLOps Pipeline — Local Runner")
    print("=" * 60)

    # 1. Check or train model
    if not MODEL_PATH.exists():
        print("[1/3] Model artifact not found. Training model now...")
        from src.model.train import train_and_save_model
        train_and_save_model()
        print("[✓] Model training completed successfully.")
    else:
        print(f"[1/3] Existing model artifact found at {MODEL_PATH}")

    # 2. Run test suite
    print("[2/3] Running pytest suite...")
    test_result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"])
    if test_result.returncode != 0:
        print("[!] Unit tests failed. Aborting server startup.")
        sys.exit(1)
    print("[✓] All tests passed!")

    # 3. Start FastAPI server
    print("[3/3] Starting FastAPI server on http://127.0.0.1:8000 ...")
    print("     Interactive API Docs: http://127.0.0.1:8000/docs")
    print("     Health Check:         http://127.0.0.1:8000/health")
    print("     Observability:        http://127.0.0.1:8000/metrics")
    print("=" * 60)

    subprocess.run([
        sys.executable,
        "-m",
        "uvicorn",
        "src.api.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--reload",
    ])


if __name__ == "__main__":
    main()
