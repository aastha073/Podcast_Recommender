"""
Collaborative filtering via Singular Value Decomposition (SVD).

How it works:
  1. Build a user × podcast interaction matrix (ratings or implicit signals)
  2. Factorize it into user latent factors (P) and item latent factors (Q)
     such that R ≈ P × Q^T
  3. At query time: take a user's latent vector, dot-product with all item
     vectors → predicted ratings for every podcast they haven't seen yet

Why SVD?
  - Handles sparsity well (most users have only listened to a tiny % of podcasts)
  - Captures latent taste dimensions (e.g. "prefers long-form science content")
  - Surprise library gives us a well-tested, fast implementation
  - Easy to swap for ALS (implicit feedback) or NMF later

Cold-start strategy (new users):
  - Fall back to popularity-based ranking
  - Once user has ≥ 5 interactions, retrain or use online update
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import cross_validate


class CollaborativeFilter:
    """
    SVD-based collaborative filtering for podcast recommendations.

    Wraps the Surprise library's SVD with helpers for batch prediction
    and integration with the rest of the pipeline.

    Example:
        cf = CollaborativeFilter(n_factors=128)
        cf.fit(train_interactions, n_users, n_podcasts)
        scores = cf.predict_for_user("user_0001", candidate_podcast_ids)
    """

    def __init__(
        self,
        n_factors: int = 128,
        n_epochs: int = 20,
        lr_all: float = 0.005,
        reg_all: float = 0.02,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all

        self._algo: Optional[SVD] = None
        self._trainset = None
        self._user_encoder: Optional[LabelEncoder] = None
        self._podcast_encoder: Optional[LabelEncoder] = None
        self._popularity_scores: Optional[dict[str, float]] = None

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        interactions: pd.DataFrame,
        user_encoder: LabelEncoder,
        podcast_encoder: LabelEncoder,
        run_cv: bool = False,
    ) -> "CollaborativeFilter":
        """
        Train the SVD model on user-podcast interactions.

        Args:
            interactions:    DataFrame with [user_id, podcast_id, rating].
            user_encoder:    Fitted LabelEncoder for user_id.
            podcast_encoder: Fitted LabelEncoder for podcast_id.
            run_cv:          If True, also run 3-fold cross-validation and
                             print RMSE/MAE (adds ~2x training time).

        Returns:
            self
        """
        self._user_encoder = user_encoder
        self._podcast_encoder = podcast_encoder

        # Compute popularity scores (used for cold-start fallback)
        counts = interactions["podcast_id"].value_counts()
        total = counts.sum()
        self._popularity_scores = (counts / total).to_dict()

        # --- Prepare Surprise Dataset ---
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(
            interactions[["user_id", "podcast_id", "rating"]], reader
        )
        trainset = data.build_full_trainset()
        self._trainset = trainset

        # --- Train SVD ---
        self._algo = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            verbose=False,
        )
        logger.info(
            f"Training SVD (n_factors={self.n_factors}, "
            f"n_epochs={self.n_epochs})..."
        )
        self._algo.fit(trainset)
        logger.info("SVD training complete.")

        if run_cv:
            logger.info("Running 3-fold cross-validation...")
            cv_results = cross_validate(
                self._algo, data, measures=["RMSE", "MAE"], cv=3, verbose=False
            )
            rmse = np.mean(cv_results["test_rmse"])
            mae = np.mean(cv_results["test_mae"])
            logger.info(f"CV RMSE: {rmse:.4f} | CV MAE: {mae:.4f}")

        return self

    # ── Predict ──────────────────────────────────────────────────────────────

    def predict_for_user(
        self,
        user_id: str,
        candidate_podcast_ids: list[str],
        is_new_user: bool = False,
    ) -> pd.DataFrame:
        """
        Predict ratings for a user across a list of candidate podcasts.

        Args:
            user_id:               User identifier.
            candidate_podcast_ids: List of podcast IDs to score.
            is_new_user:           If True, fall back to popularity scoring.

        Returns:
            DataFrame with [podcast_id, collab_score], sorted descending.
        """
        if is_new_user or not self._is_known_user(user_id):
            return self._popularity_fallback(candidate_podcast_ids)

        predictions = []
        for pod_id in candidate_podcast_ids:
            pred = self._algo.predict(uid=user_id, iid=pod_id)
            predictions.append({
                "podcast_id": pod_id,
                "collab_score": pred.est,
            })

        df = pd.DataFrame(predictions)
        # Normalize to [0, 1]
        if df["collab_score"].std() > 0:
            df["collab_score"] = (
                (df["collab_score"] - df["collab_score"].min())
                / (df["collab_score"].max() - df["collab_score"].min())
            )
        return df.sort_values("collab_score", ascending=False).reset_index(drop=True)

    def get_user_factors(self) -> tuple[list, np.ndarray]:
        """
        Return (inner_user_ids, user_factor_matrix) from the trained model.
        Useful for building user similarity clusters or visualizing taste spaces.
        """
        if self._algo is None:
            raise RuntimeError("Model not trained yet. Call fit() first.")
        inner_ids = list(range(self._trainset.n_users))
        factors = self._algo.pu  # shape: (n_users, n_factors)
        raw_ids = [self._trainset.to_raw_uid(i) for i in inner_ids]
        return raw_ids, factors

    def get_item_factors(self) -> tuple[list, np.ndarray]:
        """Return (inner_item_ids, item_factor_matrix) from the trained model."""
        if self._algo is None:
            raise RuntimeError("Model not trained yet. Call fit() first.")
        inner_ids = list(range(self._trainset.n_items))
        factors = self._algo.qi  # shape: (n_items, n_factors)
        raw_ids = [self._trainset.to_raw_iid(i) for i in inner_ids]
        return raw_ids, factors

    # ── Cold-start ───────────────────────────────────────────────────────────

    def _popularity_fallback(self, candidate_podcast_ids: list[str]) -> pd.DataFrame:
        """Return popularity-based scores for cold-start users."""
        max_pop = max(self._popularity_scores.values()) if self._popularity_scores else 1.0
        rows = [
            {
                "podcast_id": pod_id,
                "collab_score": self._popularity_scores.get(pod_id, 0.0) / max_pop,
            }
            for pod_id in candidate_podcast_ids
        ]
        return (
            pd.DataFrame(rows)
            .sort_values("collab_score", ascending=False)
            .reset_index(drop=True)
        )

    def _is_known_user(self, user_id: str) -> bool:
        """Check if a user_id exists in the training set."""
        try:
            self._trainset.to_inner_uid(user_id)
            return True
        except ValueError:
            return False

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate(self, test_interactions: pd.DataFrame) -> dict[str, float]:
        """
        Compute RMSE and MAE on a held-out test set.

        Returns:
            {"rmse": float, "mae": float}
        """
        if self._algo is None:
            raise RuntimeError("Model not trained yet.")

        preds = [
            self._algo.predict(uid=row["user_id"], iid=row["podcast_id"], r_ui=row["rating"])
            for _, row in test_interactions.iterrows()
        ]
        rmse = accuracy.rmse(preds, verbose=False)
        mae = accuracy.mae(preds, verbose=False)
        logger.info(f"Test RMSE: {rmse:.4f} | Test MAE: {mae:.4f}")
        return {"rmse": rmse, "mae": mae}

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"CollaborativeFilter saved to {path}")

    @classmethod
    def load(cls, path: str) -> "CollaborativeFilter":
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"CollaborativeFilter loaded from {path}")
        return model


# ─── Quick test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.data.loader import load_data

    dataset = load_data(use_synthetic=True)

    cf = CollaborativeFilter(n_factors=64, n_epochs=10)
    cf.fit(
        dataset.train_interactions,
        dataset.user_encoder,
        dataset.podcast_encoder,
        run_cv=False,
    )

    metrics = cf.evaluate(dataset.test_interactions)
    print(f"\nRMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f}")

    sample_user = dataset.train_interactions["user_id"].iloc[0]
    candidates = dataset.podcasts["podcast_id"].tolist()[:50]
    preds = cf.predict_for_user(sample_user, candidates)
    print(f"\nTop-5 predictions for {sample_user}:")
    print(preds.head())
