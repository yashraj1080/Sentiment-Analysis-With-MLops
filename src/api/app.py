"""
FastAPI application for serving the Sentiment Analysis MLOps pipeline.
Exposes REST endpoints for health checks, real-time predictions, batch predictions,
and observability metrics.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles

from src.api.schemas import (
    BatchSentimentRequest,
    BatchSentimentResponse,
    HealthResponse,
    MetricsResponse,
    SentimentRequest,
    SentimentResponse,
    EmotionResponse,
)
from src.model.train import load_model_pipeline, predict_sentiment, DEFAULT_MODEL_PATH, DEFAULT_METADATA_PATH
from src.model.emotion import load_emotion_model, predict_emotions, EMOTION_MODEL_PATH, EMOTION_METADATA_PATH

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


def fetch_live_headlines(feed: str, pipeline: Any, limit: int) -> list[dict[str, Any]]:
    """Fetch public Google News RSS headlines and classify their text."""
    query = 'movie reviews' if feed == 'movies' else 'news'
    url = 'https://news.google.com/rss/search?' + urllib.parse.urlencode({
        'q': query, 'hl': 'en-US', 'gl': 'US', 'ceid': 'US:en'
    })
    request = urllib.request.Request(url, headers={'User-Agent': 'SentimentStudio/1.0'})
    with urllib.request.urlopen(request, timeout=8) as response:
        root = ET.fromstring(response.read())

    items = []
    for item in root.findall('./channel/item')[:limit]:
        title = item.findtext('title', '').strip()
        description = re.sub(r'<[^>]+>', ' ', item.findtext('description', ''))
        description = re.sub(r'\s+', ' ', description).strip()
        text = f'{title}. {description}'.strip()
        if not title or not description:
            text = title or description
        if not text:
            continue
        prediction = predict_sentiment(pipeline, text[:5000])
        items.append({
            'title': title,
            'summary': description[:500],
            'url': item.findtext('link', '').strip(),
            'published': item.findtext('pubDate', '').strip(),
            'source': item.findtext('source', '').strip(),
            'sentiment': prediction['sentiment'],
            'confidence': prediction['confidence'],
        })
    return items


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to load the ML model into memory on startup."""
    logger.info("Initializing Sentiment Analysis API service...")
    
    app.state.model = None
    app.state.metadata = {}
    app.state.model_version = "1.0.0"
    app.state.emotion_model = None
    app.state.emotion_metadata = {}

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

    try:
        if EMOTION_MODEL_PATH.exists():
            app.state.emotion_model = load_emotion_model(EMOTION_MODEL_PATH)
            if EMOTION_METADATA_PATH.exists():
                app.state.emotion_metadata = json.loads(EMOTION_METADATA_PATH.read_text(encoding="utf-8"))
            logger.info("Loaded multi-label emotion model from %s", EMOTION_MODEL_PATH)
        else:
            logger.info("Emotion model is not trained yet; /predict/emotions will be unavailable")
    except Exception as exc:
        logger.error("Failed to load emotion model: %s", exc, exc_info=True)

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


STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", tags=["General"], response_class=FileResponse)
async def root():
    """Serve the interactive MLOps web dashboard."""
    if INDEX_HTML.exists():
        return FileResponse(INDEX_HTML)
    return JSONResponse(
        content={
            "service": "Sentiment Analysis MLOps Pipeline",
            "description": "Automated ML inference service running on AWS ECS Fargate",
            "status": "operational",
            "version": getattr(app.state, "model_version", "1.0.0"),
            "docs_url": "/docs",
            "health_url": "/health",
            "metrics_url": "/metrics",
        }
    )


@app.get("/api", tags=["General"])
async def api_info():
    """API programmatic metadata and documentation pointers."""
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


@app.post("/predict/emotions", response_model=EmotionResponse, tags=["Inference"])
async def predict_emotion_endpoint(payload: SentimentRequest):
    """Predict one or more fine-grained emotions expressed in a statement."""
    pipeline = getattr(app.state, "emotion_model", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Emotion model is not trained. Download GoEmotions and run `python -m src.model.emotion`.",
        )
    metadata = getattr(app.state, "emotion_metadata", {})
    result = predict_emotions(
        pipeline,
        payload.text,
        metadata.get("labels", []),
        metadata.get("thresholds", {}),
    )
    return EmotionResponse(
        text=payload.text,
        emotions=result["emotions"],
        top_emotions=result["top_emotions"],
        segments=result["segments"],
        is_neutral=result["is_neutral"],
        note=result["note"],
        model_version=metadata.get("model_version", "unknown"),
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


@app.get('/live-feed', tags=['Inference'])
async def live_feed(feed: str = 'movies', limit: int = 12):
    """Classify recent public movie-review or general-news RSS headlines."""
    if feed not in {'movies', 'news'}:
        raise HTTPException(status_code=422, detail="feed must be 'movies' or 'news'")
    pipeline = getattr(app.state, 'model', None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail='Model is not available.')
    limit = max(1, min(limit, 20))
    try:
        items = await run_in_threadpool(fetch_live_headlines, feed, pipeline, limit)
    except Exception as exc:
        logger.warning('Live RSS feed unavailable: %s', exc)
        raise HTTPException(status_code=502, detail='The live news feed is temporarily unavailable.') from exc
    return {
        'feed': feed,
        'source': 'Google News RSS',
        'analyzed_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'note': 'Sentiment is inferred from headline and short RSS summary; it does not represent audience opinion or the full article.',
        'items': items,
    }
