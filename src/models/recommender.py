"""
Unified Podcast Recommendation Pipeline

Orchestrates the full two-stage recommendation flow:
  Stage 1 — Retrieval:  Embedding model + Collaborative filter each
                         generate ~100 candidate podcasts per user.
  Stage 2 — Ranking:    LightGBM hybrid ranker re-scores and re-ranks
                         the merged candidate pool.

Also handles:
  - MLflow experiment tracking (params, metrics, artifacts)
  - Model serialization and loading
  - Single-user and batch inference

Run this file directly to train the full pipeline:
  python -m src.models.recommender

Or import and use programmatically:
  pipeline = RecommendationPipeline.load("models/")
  recs = pipeline.recommend(user_id="user_0001", query="machine learning")
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from loguru import logger

from src.config import load_project_config
from src.data.loader import PodcastDataset, load_data
from src.models.collaborative import CollaborativeFilter
from src.models.embedder import PodcastEmbedder
from src.models.ranker import HybridRanker


class RecommendationPipeline:
    """
    End-to-end podcast recommendation pipeline.

    Combines:
      - PodcastEmbedder   (semantic content retrieval via Sentence-BERT + FAISS)
      - CollaborativeFilter (user taste modeling via SVD)
      - HybridRanker       (final re-ranking via LightGBM)

    Designed to be:
      - Trainable from a single .fit() call
      - Serializable for deployment
      - Tracked in MLflow for experiment comparison
    """

    def __init__(
        self,
        # Embedder config
        embedder_model: str = "all-MiniLM-L6-v2",
        embedding_batch_size: int = 64,
        # Collaborative filter config
        n_factors: int = 128,
        n_epochs: int = 20,
        # Ranker config
        ranker_n_estimators: int = 300,
        ranker_learning_rate: float = 0.05,
        ranker_num_leaves: int = 63,
        # Recommendation config
        candidate_pool_size: int = 100,
        top_k: int = 10,
        diversity_penalty: float = 0.1,
        # MLflow config
        mlflow_tracking_uri: str = "http://localhost:5000",
        mlflow_experiment: str = "podcast-recsys",
    ):
        self.embedder_model = embedder_model
        self.embedding_batch_size = embedding_batch_size
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.ranker_n_estimators = ranker_n_estimators
        self.ranker_learning_rate = ranker_learning_rate
        self.ranker_num_leaves = ranker_num_leaves
        self.candidate_pool_size = candidate_pool_size
        self.top_k = top_k
        self.diversity_penalty = diversity_penalty
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_experiment = mlflow_experiment

        self.embedder: Optional[PodcastEmbedder] = None
        self.cf_model: Optional[CollaborativeFilter] = None
        self.ranker: Optional[HybridRanker] = None
        self._podcasts: Optional[pd.DataFrame] = None

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(
        self,
        dataset: PodcastDataset,
        use_mlflow: bool = True,
        run_name: Optional[str] = None,
    ) -> "RecommendationPipeline":
        """
        Train all three stages and log everything to MLflow.

        Args:
            dataset:     PodcastDataset from load_data().
            use_mlflow:  If True, log params/metrics/artifacts to MLflow.
                         Set to False for quick local testing.
            run_name:    MLflow run name. Auto-generated if None.

        Returns:
            self
        """
        self._podcasts = dataset.podcasts

        if use_mlflow:
            try:
                mlflow.set_tracking_uri(self.mlflow_tracking_uri)
                mlflow.set_experiment(self.mlflow_experiment)
            except Exception as e:
                logger.warning(f"MLflow connection failed: {e}. Proceeding without tracking.")
                use_mlflow = False

        run_ctx = mlflow.start_run(run_name=run_name) if use_mlflow else _NoOpContext()

        with run_ctx:
            if use_mlflow:
                self._log_params()

            # ── Stage 1a: Embedder ────────────────────────────────────────
            logger.info("=== Stage 1a: Training Embedder ===")
            t0 = time.time()
            self.embedder = PodcastEmbedder(
                model_name=self.embedder_model,
                batch_size=self.embedding_batch_size,
            )
            self.embedder.fit(dataset.podcasts)
            embedder_time = time.time() - t0
            logger.info(f"Embedder ready in {embedder_time:.1f}s")

            # ── Stage 1b: Collaborative Filter ───────────────────────────
            logger.info("=== Stage 1b: Training Collaborative Filter ===")
            t0 = time.time()
            self.cf_model = CollaborativeFilter(
                n_factors=self.n_factors,
                n_epochs=self.n_epochs,
            )
            self.cf_model.fit(
                dataset.train_interactions,
                dataset.user_encoder,
                dataset.podcast_encoder,
            )
            cf_metrics = self.cf_model.evaluate(dataset.test_interactions)
            cf_time = time.time() - t0
            logger.info(f"CF model ready in {cf_time:.1f}s | RMSE: {cf_metrics['rmse']:.4f}")

            # ── Stage 2: Ranker ───────────────────────────────────────────
            logger.info("=== Stage 2: Training LightGBM Ranker ===")
            t0 = time.time()
            self.ranker = HybridRanker(
                n_estimators=self.ranker_n_estimators,
                learning_rate=self.ranker_learning_rate,
                num_leaves=self.ranker_num_leaves,
            )
            self.ranker.fit(
                interactions=dataset.train_interactions,
                podcasts=dataset.podcasts,
            )
            ndcg_at_10 = self.ranker.evaluate_ndcg(
                dataset.test_interactions, dataset.podcasts, k=10
            )
            ranker_time = time.time() - t0
            logger.info(f"Ranker ready in {ranker_time:.1f}s | NDCG@10: {ndcg_at_10:.4f}")

            # ── Log metrics ───────────────────────────────────────────────
            if use_mlflow:
                mlflow.log_metrics({
                    "cf_rmse": cf_metrics["rmse"],
                    "cf_mae": cf_metrics["mae"],
                    "ndcg_at_10": ndcg_at_10,
                    "embedder_fit_time_s": embedder_time,
                    "cf_fit_time_s": cf_time,
                    "ranker_fit_time_s": ranker_time,
                    "n_podcasts": len(dataset.podcasts),
                    "n_users": dataset.interactions["user_id"].nunique(),
                    "n_train_interactions": len(dataset.train_interactions),
                })
                logger.info("Metrics logged to MLflow.")

        logger.info("Pipeline training complete.")
        return self

    # ── Inference ────────────────────────────────────────────────────────────

    def recommend(
        self,
        user_id: str,
        query: str = "",
        user_history: Optional[list[str]] = None,
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generate recommendations for a single user.

        Args:
            user_id:      User identifier. If unknown, falls back to popularity.
            query:        Free-text interest description (e.g. "AI and science").
                          If empty, uses collaborative filter only.
            user_history: List of already-listened podcast IDs to exclude.
            top_k:        Override default top_k if provided.

        Returns:
            DataFrame [podcast_id, title, category, description,
                       embedding_score, collab_score, ranker_score]
            sorted by ranker_score descending.
        """
        self._assert_trained()
        k = top_k or self.top_k
        exclude = set(user_history or [])

        # ── Retrieval: Embedding candidates ──────────────────────────────
        if query:
            emb_candidates = self.embedder.query(
                query,
                top_k=self.candidate_pool_size,
                exclude_ids=list(exclude),
            )
        else:
            # No query — sample broad set of podcasts for CF to score
            all_ids = self._podcasts["podcast_id"].tolist()
            emb_candidates = self._podcasts[
                ~self._podcasts["podcast_id"].isin(exclude)
            ].copy()
            emb_candidates["embedding_score"] = 0.0

        # ── Retrieval: Collaborative filter candidates ────────────────────
        candidate_ids = emb_candidates["podcast_id"].tolist()
        cf_scores = self.cf_model.predict_for_user(user_id, candidate_ids)

        # ── Merge candidate pools ─────────────────────────────────────────
        candidates = emb_candidates.merge(
            cf_scores[["podcast_id", "collab_score"]],
            on="podcast_id",
            how="left",
        )
        candidates["collab_score"] = candidates["collab_score"].fillna(0.0)

        # Merge podcast metadata for ranker features
        meta_cols = ["podcast_id", "avg_duration_min", "total_episodes",
                     "play_count", "release_year", "category"]
        available = [c for c in meta_cols if c in self._podcasts.columns]
        candidates = candidates.merge(
            self._podcasts[available], on="podcast_id", how="left", suffixes=("", "_meta")
        )
        # Resolve duplicate category column if both exist
        if "category_meta" in candidates.columns:
            candidates["category"] = candidates["category"].fillna(candidates["category_meta"])
            candidates.drop(columns=["category_meta"], inplace=True)

        # ── Ranking ───────────────────────────────────────────────────────
        ranked = self.ranker.rank(
            user_id=user_id,
            candidates=candidates,
            user_history=list(exclude),
            top_k=k,
            diversity_penalty=self.diversity_penalty,
        )

        return ranked

    # ── Batch inference ───────────────────────────────────────────────────────

    def recommend_batch(
        self,
        user_ids: list[str],
        queries: Optional[dict[str, str]] = None,
        top_k: Optional[int] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Generate recommendations for multiple users.

        Args:
            user_ids: List of user IDs.
            queries:  Optional {user_id: query_text} mapping.
            top_k:    Number of recommendations per user.

        Returns:
            {user_id: recommendations_df} mapping.
        """
        results = {}
        queries = queries or {}
        for uid in user_ids:
            results[uid] = self.recommend(
                user_id=uid,
                query=queries.get(uid, ""),
                top_k=top_k,
            )
        return results

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _assert_trained(self) -> None:
        if any(m is None for m in [self.embedder, self.cf_model, self.ranker]):
            raise RuntimeError(
                "Pipeline not trained. Call fit() or load() first."
            )

    def _log_params(self) -> None:
        mlflow.log_params({
            "embedder_model": self.embedder_model,
            "n_factors": self.n_factors,
            "n_epochs": self.n_epochs,
            "ranker_n_estimators": self.ranker_n_estimators,
            "ranker_learning_rate": self.ranker_learning_rate,
            "ranker_num_leaves": self.ranker_num_leaves,
            "candidate_pool_size": self.candidate_pool_size,
            "top_k": self.top_k,
            "diversity_penalty": self.diversity_penalty,
        })

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, model_dir: str) -> None:
        """Save all pipeline components to disk."""
        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.embedder.save(str(path / "embedder"))
        self.cf_model.save(str(path / "cf_model.pkl"))
        self.ranker.save(str(path / "ranker.pkl"))
        self._podcasts.to_parquet(path / "podcasts.parquet")
        logger.info(f"Pipeline saved to {model_dir}")

    @classmethod
    def load(cls, model_dir: str) -> "RecommendationPipeline":
        """Load a previously saved pipeline."""
        path = Path(model_dir)
        pipeline = cls()
        pipeline.embedder = PodcastEmbedder.load(str(path / "embedder"))
        pipeline.cf_model = CollaborativeFilter.load(str(path / "cf_model.pkl"))
        pipeline.ranker = HybridRanker.load(str(path / "ranker.pkl"))
        pipeline._podcasts = pd.read_parquet(path / "podcasts.parquet")
        logger.info(f"Pipeline loaded from {model_dir}")
        return pipeline


# ─── No-op context manager (when MLflow is off) ───────────────────────────────

class _NoOpContext:
    def __enter__(self): return self
    def __exit__(self, *args): pass


# ─── CLI entrypoint ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the podcast recommendation pipeline.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--source", choices=["synthetic", "csv", "rss", "kaggle"], default=None)
    parser.add_argument("--synthetic", action="store_true", default=False,
                        help="Use synthetic data (overrides --source)")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Disable MLflow tracking")
    parser.add_argument("--n-factors", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--save-dir", type=str, default="models/pipeline")
    args = parser.parse_args()

    cfg = load_project_config(args.config)
    data_cfg = cfg.get("data", {})
    source = args.source or data_cfg.get("source", "synthetic")
    if args.synthetic:
        source = "synthetic"

    logger.info("Loading data...")
    podcasts_path = data_cfg.get("podcasts_path")
    interactions_path = data_cfg.get("interactions_path")
    if source == "rss":
        podcasts_path = data_cfg.get("rss_processed_podcasts_path", "data/processed/podcasts.parquet")
        interactions_path = data_cfg.get("rss_processed_interactions_path", "data/processed/interactions.parquet")
    if source == "kaggle":
        podcasts_path = data_cfg.get("kaggle_processed_podcasts_path", "data/processed/podcasts.parquet")
        interactions_path = data_cfg.get("kaggle_processed_interactions_path", "data/processed/interactions.parquet")

    dataset = load_data(
        podcasts_path=podcasts_path,
        interactions_path=interactions_path,
        test_size=float(data_cfg.get("test_size", 0.2)),
        min_interactions=int(data_cfg.get("min_interactions", 5)),
        random_seed=int(data_cfg.get("random_seed", 42)),
        use_synthetic=(source == "synthetic"),
        source=source,
    )

    logger.info("Building pipeline...")
    pipeline = RecommendationPipeline(
        n_factors=args.n_factors,
        n_epochs=args.n_epochs,
        ranker_n_estimators=args.n_estimators,
    )

    pipeline.fit(dataset, use_mlflow=not args.no_mlflow)
    pipeline.save(args.save_dir)

    # --- Demo: recommend for a sample user ---
    sample_user = dataset.interactions["user_id"].iloc[0]
    logger.info(f"\nDemo recommendations for {sample_user}:")
    recs = pipeline.recommend(
        user_id=sample_user,
        query="technology and artificial intelligence",
        top_k=5,
    )
    print(recs[["title", "category", "ranker_score"]].to_string(index=False))
