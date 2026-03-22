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
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder


class CollaborativeFilter:
    """
    SVD-based collaborative filtering for podcast recommendations.

    Uses scikit-learn's TruncatedSVD for matrix factorization with helpers 
    for batch prediction and integration with the rest of the pipeline.

    Example:
        cf = CollaborativeFilter(n_factors=128)
        cf.fit(train_interactions, user_encoder, podcast_encoder)
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

        self._svd: Optional[TruncatedSVD] = None
        self._user_factors: Optional[np.ndarray] = None
        self._item_factors: Optional[np.ndarray] = None
        self._user_encoder: Optional[LabelEncoder] = None
        self._podcast_encoder: Optional[LabelEncoder] = None
        self._popularity_scores: Optional[dict[str, float]] = None
        self._user_means: Optional[np.ndarray] = None
        self._user_ids: Optional[list] = None
        self._podcast_ids: Optional[list] = None

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
            run_cv:          If True, also run simple train/test split validation.

        Returns:
            self
        """
        self._user_encoder = user_encoder
        self._podcast_encoder = podcast_encoder

        # Compute popularity scores (used for cold-start fallback)
        counts = interactions["podcast_id"].value_counts()
        total = counts.sum()
        self._popularity_scores = (counts / total).to_dict()

        # --- Build rating matrix ---
        n_users = len(user_encoder.classes_)
        n_podcasts = len(podcast_encoder.classes_)
        
        # Map user_id and podcast_id to indices
        user_indices = user_encoder.transform(interactions["user_id"])
        podcast_indices = podcast_encoder.transform(interactions["podcast_id"])
        ratings = interactions["rating"].values
        
        # Create sparse CSR matrix (user × podcast)
        rating_matrix = csr_matrix(
            (ratings, (user_indices, podcast_indices)),
            shape=(n_users, n_podcasts),
            dtype=np.float32
        )
        
        # Store user and podcast ID mappings
        self._user_ids = list(user_encoder.classes_)
        self._podcast_ids = list(podcast_encoder.classes_)
        
        # Compute user means for centering
        self._user_means = np.array(rating_matrix.mean(axis=1)).flatten()
        
        # Center the matrix
        centered_matrix = rating_matrix.copy().astype(np.float32)
        centered_matrix.data -= self._user_means[rating_matrix.nonzero()[0]]
        
        # --- Train TruncatedSVD ---
        logger.info(
            f"Training SVD (n_factors={self.n_factors})..."
        )
        self._svd = TruncatedSVD(n_components=self.n_factors, random_state=42)
        self._svd.fit(centered_matrix)
        
        # Get factors
        self._user_factors = self._svd.fit_transform(centered_matrix)  # (n_users, n_factors)
        self._item_factors = self._svd.components_.T  # (n_podcasts, n_factors)
        
        logger.info("SVD training complete.")

        if run_cv:
            logger.info("Running train/test validation...")
            from sklearn.model_selection import train_test_split
            train_idx, test_idx = train_test_split(
                np.arange(len(interactions)), test_size=0.2, random_state=42
            )
            test_data = interactions.iloc[test_idx]
            
            metrics = self.evaluate(test_data)
            logger.info(f"Validation RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f}")

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

        # Get user factor
        try:
            user_idx = self._user_encoder.transform([user_id])[0]
        except:
            return self._popularity_fallback(candidate_podcast_ids)
        
        user_factor = self._user_factors[user_idx]  # (n_factors,)
        
        predictions = []
        for pod_id in candidate_podcast_ids:
            try:
                pod_idx = self._podcast_encoder.transform([pod_id])[0]
                item_factor = self._item_factors[pod_idx]  # (n_factors,)
                # Predict: user_mean + dot product of factors
                pred_score = self._user_means[user_idx] + np.dot(user_factor, item_factor)
                predictions.append({
                    "podcast_id": pod_id,
                    "collab_score": float(pred_score),
                })
            except:
                # Unknown podcast, use 0
                predictions.append({
                    "podcast_id": pod_id,
                    "collab_score": 0.0,
                })

        df = pd.DataFrame(predictions)
        # Normalize to [0, 1]
        if len(df) > 0 and df["collab_score"].std() > 0:
            df["collab_score"] = (
                (df["collab_score"] - df["collab_score"].min())
                / (df["collab_score"].max() - df["collab_score"].min())
            )
        return df.sort_values("collab_score", ascending=False).reset_index(drop=True)

    def get_user_factors(self) -> tuple[list, np.ndarray]:
        """
        Return (user_ids, user_factor_matrix) from the trained model.
        Useful for building user similarity clusters or visualizing taste spaces.
        """
        if self._user_factors is None:
            raise RuntimeError("Model not trained yet. Call fit() first.")
        return self._user_ids, self._user_factors

    def get_item_factors(self) -> tuple[list, np.ndarray]:
        """Return (podcast_ids, item_factor_matrix) from the trained model."""
        if self._item_factors is None:
            raise RuntimeError("Model not trained yet. Call fit() first.")
        return self._podcast_ids, self._item_factors

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
            self._user_encoder.transform([user_id])
            return True
        except:
            return False

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate(self, test_interactions: pd.DataFrame) -> dict[str, float]:
        """
        Compute RMSE and MAE on a held-out test set.

        Returns:
            {"rmse": float, "mae": float}
        """
        if self._user_factors is None:
            raise RuntimeError("Model not trained yet.")

        predictions = []
        for _, row in test_interactions.iterrows():
            pred_df = self.predict_for_user(
                row["user_id"], 
                [row["podcast_id"]], 
                is_new_user=False
            )
            if len(pred_df) > 0:
                pred_score = pred_df["collab_score"].iloc[0]
            else:
                pred_score = self._user_means.mean() if self._user_means is not None else 0.5
            predictions.append(pred_score)
        
        predictions = np.array(predictions)
        actual = test_interactions["rating"].values
        
        rmse = np.sqrt(np.mean((predictions - actual) ** 2))
        mae = np.mean(np.abs(predictions - actual))
        
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
