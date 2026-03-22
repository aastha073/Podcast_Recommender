"""
FastAPI serving layer for the podcast recommendation system.

Endpoints:
  GET  /health              — liveness + readiness check
  GET  /metrics             — model metadata and serving stats
  POST /recommend           — get recommendations for a user
  POST /recommend/batch     — batch recommendations for multiple users
  GET  /similar/{podcast_id} — find podcasts similar to a given one

Run locally:
  uvicorn src.api.main:app --reload --port 8000

Then open: http://localhost:8000/docs  (Swagger UI, auto-generated)
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

from src.models.recommender import RecommendationPipeline


# ─── Global state ─────────────────────────────────────────────────────────────

pipeline: Optional[RecommendationPipeline] = None
_start_time = time.time()
_request_count = 0


# ─── Startup / shutdown ───────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model pipeline on startup, release on shutdown."""
    global pipeline
    model_dir = Path("models/pipeline")

    if model_dir.exists():
        logger.info(f"Loading pipeline from {model_dir}...")
        pipeline = RecommendationPipeline.load(str(model_dir))
        logger.info("Pipeline loaded. API ready.")
    else:
        # Train fresh on synthetic data if no saved model exists
        logger.warning(f"No saved model at {model_dir}. Training on synthetic data...")
        from src.data.loader import load_data
        dataset = load_data(use_synthetic=True)
        pipeline = RecommendationPipeline(
            n_factors=64, n_epochs=10, ranker_n_estimators=100
        )
        pipeline.fit(dataset, use_mlflow=False)
        pipeline.save(str(model_dir))
        logger.info("Pipeline trained and saved.")

    yield

    logger.info("Shutting down API.")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Podcast Recommendation API",
    description="AI-powered podcast recommendations using semantic embeddings + collaborative filtering.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response schemas ───────────────────────────────────────────────

class RecommendRequest(BaseModel):
    user_id: str = Field(..., example="user_0001", description="User identifier")
    query: str = Field(
        default="",
        example="technology and artificial intelligence",
        description="Free-text interest description. Leave empty for pure collaborative filtering.",
    )
    top_k: int = Field(default=10, ge=1, le=50, description="Number of recommendations")
    exclude_podcast_ids: list[str] = Field(
        default_factory=list,
        description="Podcast IDs to exclude (e.g. already listened)",
    )


class BatchRecommendRequest(BaseModel):
    user_ids: list[str] = Field(..., min_length=1, max_length=100)
    query: str = Field(default="")
    top_k: int = Field(default=10, ge=1, le=50)


class PodcastResult(BaseModel):
    podcast_id: str
    title: str
    category: str
    description: str
    ranker_score: float


class RecommendResponse(BaseModel):
    user_id: str
    recommendations: list[PodcastResult]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float
    requests_served: int


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    """Liveness + readiness check. Returns 200 when the model is loaded and ready."""
    return HealthResponse(
        status="ok" if pipeline is not None else "loading",
        model_loaded=pipeline is not None,
        uptime_seconds=round(time.time() - _start_time, 1),
        requests_served=_request_count,
    )


@app.get("/metrics", tags=["ops"])
def metrics():
    """Model metadata and serving statistics."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return {
        "model": {
            "embedder": pipeline.embedder_model,
            "n_factors": pipeline.n_factors,
            "ranker_estimators": pipeline.ranker_n_estimators,
            "top_k_default": pipeline.top_k,
            "diversity_penalty": pipeline.diversity_penalty,
        },
        "serving": {
            "uptime_seconds": round(time.time() - _start_time, 1),
            "requests_served": _request_count,
        },
    }


@app.post("/recommend", response_model=RecommendResponse, tags=["recommendations"])
def recommend(req: RecommendRequest):
    """
    Get personalized podcast recommendations for a user.

    - Combine a `user_id` with an optional free-text `query` for best results.
    - Omit `query` to get pure collaborative-filter recommendations.
    - New/unknown users fall back to popularity-based ranking.
    """
    global _request_count
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    t0 = time.time()
    try:
        recs_df = pipeline.recommend(
            user_id=req.user_id,
            query=req.query,
            user_history=req.exclude_podcast_ids,
            top_k=req.top_k,
        )
    except Exception as e:
        logger.error(f"Recommendation error for user {req.user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = (time.time() - t0) * 1000
    _request_count += 1

    recommendations = []
    for _, row in recs_df.iterrows():
        recommendations.append(PodcastResult(
            podcast_id=str(row.get("podcast_id", "")),
            title=str(row.get("title", "")),
            category=str(row.get("category", "")),
            description=str(row.get("description", ""))[:300],
            ranker_score=float(row.get("ranker_score", 0.0)),
        ))

    logger.info(
        f"recommend | user={req.user_id} query='{req.query}' "
        f"k={req.top_k} latency={latency_ms:.1f}ms"
    )
    return RecommendResponse(
        user_id=req.user_id,
        recommendations=recommendations,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/recommend/batch", tags=["recommendations"])
def recommend_batch(req: BatchRecommendRequest):
    """
    Batch recommendations for multiple users in a single call.
    Returns a dict of {user_id: [recommendations]}.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    t0 = time.time()
    queries = {uid: req.query for uid in req.user_ids}
    results = pipeline.recommend_batch(req.user_ids, queries=queries, top_k=req.top_k)

    output = {}
    for uid, df in results.items():
        output[uid] = [
            {
                "podcast_id": str(row.get("podcast_id", "")),
                "title": str(row.get("title", "")),
                "category": str(row.get("category", "")),
                "ranker_score": float(row.get("ranker_score", 0.0)),
            }
            for _, row in df.iterrows()
        ]

    latency_ms = (time.time() - t0) * 1000
    logger.info(f"batch_recommend | n_users={len(req.user_ids)} latency={latency_ms:.1f}ms")
    return {"results": output, "latency_ms": round(latency_ms, 2)}


@app.get("/similar/{podcast_id}", tags=["recommendations"])
def similar_podcasts(
    podcast_id: str,
    top_k: int = Query(default=10, ge=1, le=50),
):
    """
    Find podcasts semantically similar to a given podcast_id.
    Uses the embedding index — no user context needed.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Look up the podcast's description and use it as a query
    podcasts = pipeline._podcasts
    match = podcasts[podcasts["podcast_id"] == podcast_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"podcast_id '{podcast_id}' not found.")

    row = match.iloc[0]
    query_text = f"{row['title']} {row.get('category', '')} {row.get('description', '')}"

    results = pipeline.embedder.query(
        query_text,
        top_k=top_k + 1,  # +1 because the podcast itself will appear
        exclude_ids=[podcast_id],
    )

    return {
        "source_podcast": {"podcast_id": podcast_id, "title": str(row["title"])},
        "similar": results[["podcast_id", "title", "category", "embedding_score"]]
        .head(top_k)
        .to_dict(orient="records"),
    }
