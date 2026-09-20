"""
FastAPI application for serving the Sentiment Analysis MLOps pipeline.
Exposes REST endpoints for health checks, real-time predictions, batch predictions,
and observability metrics.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.schemas import (
    BatchSentimentRequest,
    BatchSentimentResponse,
    HealthResponse,
    MetricsResponse,
    SentimentRequest,
    SentimentResponse,
)
from src.model.train import load_model_pipeline, predict_sentiment, DEFAULT_MODEL_PATH, DEFAULT_METADATA_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sentiment_api")

START_TIME = time.time()


class MetricsTracker:
    """In-memory metrics tracker for API observability."""

    def __init__(self) -> None:
        self.total_requests: int = 0
        self.total_predictions: int = 0
        self.positive_predictions: int = 0
        self.negative_predictions: int = 0
        self.total_latency_ms: float = 0.0

    def record_prediction(self, sentiment: str, latency_ms: float) -> None:
        self.total_predictions += 1
        self.total_latency_ms += latency_ms
        if sentiment.lower() == "positive":
            self.positive_predictions += 1
        elif sentiment.lower() == "negative":
            self.negative_predictions += 1

    @property
    def average_latency_ms(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return round(self.total_latency_ms / self.total_predictions, 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "total_predictions": self.total_predictions,
            "positive_predictions": self.positive_predictions,
            "negative_predictions": self.negative_predictions,
            "average_latency_ms": self.average_latency_ms,
            "uptime_seconds": round(time.time() - START_TIME, 2),
        }


metrics_tracker = MetricsTracker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to load the ML model into memory on startup."""
    logger.info("Initializing Sentiment Analysis API service...")
    
    app.state.model = None
    app.state.metadata = {}
    app.state.model_version = "1.0.0"

    try:
        if DEFAULT_MODEL_PATH.exists():
            app.state.model = load_model_pipeline(DEFAULT_MODEL_PATH)
            logger.info(f"Loaded ML model pipeline from {DEFAULT_MODEL_PATH}")
        else:
            logger.warning(
                f"Model artifact not found at {DEFAULT_MODEL_PATH}. "
                "The API will start in degraded mode until the model is trained."
            )

        if DEFAULT_METADATA_PATH.exists():
            with open(DEFAULT_METADATA_PATH, "r", encoding="utf-8") as f:
                app.state.metadata = json.load(f)
                app.state.model_version = app.state.metadata.get("model_version", "1.0.0")
            logger.info(f"Loaded model metadata: version {app.state.model_version}")

    except Exception as exc:
        logger.error(f"Failed to load model pipeline: {exc}", exc_info=True)

    yield

    logger.info("Shutting down Sentiment Analysis API service...")


app = FastAPI(
    title="Sentiment Analysis MLOps REST API",
    description=(
        "Production-ready FastAPI service for NLP sentiment analysis classification, "
        "deployed via Docker, AWS ECS Fargate, and GitHub Actions CI/CD."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def count_requests_middleware(request, call_next):
    """Track request volume for observability."""
    metrics_tracker.total_requests += 1
    response = await call_next(request)
    return response


@app.get("/", tags=["General"])
async def root():
    """API root overview and documentation pointer."""
    return {
        "service": "Sentiment Analysis MLOps Pipeline",
        "description": "Automated ML inference service running on AWS ECS Fargate",
        "status": "operational",
        "version": getattr(app.state, "model_version", "1.0.0"),
        "docs_url": "/docs",
        "health_url": "/health",
        "metrics_url": "/metrics",
    }


@app.get("/health", response_model=HealthResponse, tags=["Observability"])
async def health_check():
    """Health check endpoint used by AWS Application Load Balancer and ECS task definitions."""
    model_loaded = getattr(app.state, "model", None) is not None
    uptime = round(time.time() - START_TIME, 2)
    env = os.getenv("APP_ENV", "development")
    
    if not model_loaded:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "degraded",
                "model_loaded": False,
                "model_version": "none",
                "uptime_seconds": uptime,
                "environment": env,
                "detail": "Model artifact not loaded. Run `python -m src.model.train` to train the model.",
            },
        )

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_version=getattr(app.state, "model_version", "1.0.0"),
        uptime_seconds=uptime,
        environment=env,
    )


@app.post(
    "/predict",
    response_model=SentimentResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
async def predict_single(payload: SentimentRequest):
    """Run real-time sentiment analysis on a single text string."""
    pipeline = getattr(app.state, "model", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not available. Please ensure model training has completed.",
        )

    t0 = time.perf_counter()
    result = predict_sentiment(pipeline, payload.text)
    latency_ms = round((time.perf_counter() - t0) * 1000, 3)

    metrics_tracker.record_prediction(result["sentiment"], latency_ms)

    return SentimentResponse(
        text=payload.text,
        sentiment=result["sentiment"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        latency_ms=latency_ms,
        model_version=getattr(app.state, "model_version", "1.0.0"),
    )


@app.post(
    "/predict/batch",
    response_model=BatchSentimentResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
async def predict_batch(payload: BatchSentimentRequest):
    """Run batch sentiment analysis on multiple text strings (up to 100)."""
    pipeline = getattr(app.state, "model", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not available. Please ensure model training has completed.",
        )

    t0 = time.perf_counter()
    results = []

    for text in payload.texts:
        sub_t0 = time.perf_counter()
        res = predict_sentiment(pipeline, text)
        sub_latency = round((time.perf_counter() - sub_t0) * 1000, 3)
        metrics_tracker.record_prediction(res["sentiment"], sub_latency)

        results.append(
            SentimentResponse(
                text=text,
                sentiment=res["sentiment"],
                confidence=res["confidence"],
                probabilities=res["probabilities"],
                latency_ms=sub_latency,
                model_version=getattr(app.state, "model_version", "1.0.0"),
            )
        )

    total_latency_ms = round((time.perf_counter() - t0) * 1000, 3)

    return BatchSentimentResponse(
        results=results,
        total_processed=len(results),
        total_latency_ms=total_latency_ms,
        model_version=getattr(app.state, "model_version", "1.0.0"),
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Observability"])
async def get_metrics():
    """Retrieve runtime prediction statistics and operational metrics."""
    return MetricsResponse(**metrics_tracker.to_dict())
