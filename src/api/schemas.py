"""
Pydantic schemas for request and response validation in the Sentiment Analysis REST API.
"""

from __future__ import annotations

from typing import Dict, List
from pydantic import BaseModel, Field


class SentimentRequest(BaseModel):
    """Single sentiment analysis prediction request."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Raw text review or feedback to classify",
        examples=["This movie was an absolute masterpiece! Brilliant acting."],
    )


class SentimentResponse(BaseModel):
    """Single sentiment prediction output with confidence and latency metrics."""
    text: str = Field(description="Original input text")
    sentiment: str = Field(description="Predicted sentiment label ('positive' or 'negative')")
    confidence: float = Field(description="Probability confidence score between 0.0 and 1.0")
    probabilities: Dict[str, float] = Field(
        description="Class-wise probability distribution (e.g., {'positive': 0.94, 'negative': 0.06})"
    )
    latency_ms: float = Field(description="Inference execution time in milliseconds")
    model_version: str = Field(description="Version of the model serving predictions")


class BatchSentimentRequest(BaseModel):
    """Batch sentiment analysis prediction request."""
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of text strings to analyze (max 100 per batch)",
        examples=[
            [
                "Fantastic service and great quality!",
                "Terrible experience, broke on the first day.",
            ]
        ],
    )


class BatchSentimentResponse(BaseModel):
    """Batch sentiment prediction output."""
    results: List[SentimentResponse] = Field(description="List of sentiment prediction results")
    total_processed: int = Field(description="Total number of reviews processed")
    total_latency_ms: float = Field(description="Total batch inference latency in milliseconds")
    model_version: str = Field(description="Model version used for batch inference")


class HealthResponse(BaseModel):
    """System and model health check response."""
    status: str = Field(default="healthy", description="Overall health status")
    model_loaded: bool = Field(description="Flag indicating if the ML pipeline is loaded in memory")
    model_version: str = Field(description="Version identifier of the loaded model")
    uptime_seconds: float = Field(description="Number of seconds the API server has been running")
    environment: str = Field(description="Runtime environment (e.g., development, production)")


class MetricsResponse(BaseModel):
    """Observability metrics tracking request count, sentiment distribution, and latency."""
    total_requests: int = Field(description="Total API calls received across all endpoints")
    total_predictions: int = Field(description="Total single and batch predictions served")
    positive_predictions: int = Field(description="Count of positive predictions returned")
    negative_predictions: int = Field(description="Count of negative predictions returned")
    average_latency_ms: float = Field(description="Rolling average inference latency in milliseconds")
    uptime_seconds: float = Field(description="API service uptime in seconds")
