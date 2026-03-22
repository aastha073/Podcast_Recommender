"""
Hybrid LightGBM ranker — the final stage of the recommendation pipeline.

The two-stage architecture:
  Stage 1 (Retrieval): Embedding model + collaborative filter each produce
                       a pool of ~100 candidate podcasts per user.
  Stage 2 (Ranking):   This model re-ranks that pool using richer features —
                       combining the two retrieval scores with content features,
                       user features, and podcast metadata.

Why LightGBM for ranking?
  - Handles mixed feature types (floats, ints, categoricals) natively
  - Fast training and inference
  - Interpretable (SHAP values, feature importances)
  - Can be trained with pairwise ranking loss (LambdaRank) or pointwise

Feature groups:
  - Retrieval scores:   embedding_score, collab_score (from Stage 1)
  - Podcast features:   avg_duration_min, total_episodes, play_count, release_year
  - User-podcast:       category match (does user prefer this category?)
  - Popularity signals: normalized play count, rating count
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import LabelEncoder


FEATURE_COLS = [
    # Retrieval scores (most important features for ranking)
    "embedding_score",
    "collab_score",
    # Podcast metadata
    "avg_duration_min",
    "total_episodes",
    "play_count_norm",
    "release_year_norm",
    # Interaction-based
    "category_match",
    "popularity_score",
]


class HybridRanker:
    """
    LightGBM model that re-ranks candidate podcasts per user.

    Training:
        For each (user, podcast) pair in the training set, we build a feature
        vector and use the observed rating as the target. The model learns which
        combination of features predicts a high rating.

    Inference:
        Given a user and a list of candidate podcasts (from the retrieval stage),
        predict a score for each and return them sorted descending.

    Example:
        ranker = HybridRanker()
        ranker.fit(train_df, podcasts_df)
        ranked = ranker.rank(user_id, candidates_df, user_history)
    """

    def __init__(
        self,
        num_leaves: int = 63,
        learning_rate: float = 0.05,
        n_estimators: int = 300,
        min_child_samples: int = 20,
        feature_fraction: float = 0.8,
        bagging_fraction: float = 0.8,
        bagging_freq: int = 5,
        random_seed: int = 42,
    ):
        self.params = {
            "objective": "regression",          # Pointwise; swap "lambdarank" for pairwise
            "metric": "mse",  # Changed from ndcg to mse for regression with float labels
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "min_child_samples": min_child_samples,
            "feature_fraction": feature_fraction,
            "bagging_fraction": bagging_fraction,
            "bagging_freq": bagging_freq,
            "seed": random_seed,
            "verbose": -1,
            "num_threads": 1,  # Disable parallelism to avoid fork issues on macOS
        }
        self._model: Optional[lgb.LGBMRegressor] = None
        self._user_category_prefs: dict[str, dict[str, float]] = {}
        self._play_count_max: float = 1.0
        self._release_year_min: float = 2010.0
        self._release_year_max: float = 2024.0

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        interactions: pd.DataFrame,
        podcasts: pd.DataFrame,
        embedding_scores: Optional[pd.DataFrame] = None,
        collab_scores: Optional[pd.DataFrame] = None,
    ) -> "HybridRanker":
        """
        Train the ranker on historical interaction data.

        Args:
            interactions:     Train interactions [user_id, podcast_id, rating, ...].
            podcasts:         Podcast metadata DataFrame.
            embedding_scores: Optional pre-computed embedding scores
                              [user_id, podcast_id, embedding_score].
            collab_scores:    Optional pre-computed collab scores
                              [user_id, podcast_id, collab_score].

        Returns:
            self
        """
        logger.info("Building ranker training features...")

        # --- Precompute user category preferences ---
        merged = interactions.merge(
            podcasts[["podcast_id", "category"]], on="podcast_id", how="left"
        )
        for user_id, grp in merged.groupby("user_id"):
            cat_counts = grp["category"].value_counts(normalize=True)
            self._user_category_prefs[user_id] = cat_counts.to_dict()

        # --- Normalize global features ---
        self._play_count_max = podcasts["play_count"].max()
        self._release_year_min = podcasts["release_year"].min()
        self._release_year_max = podcasts["release_year"].max()

        # --- Build feature matrix ---
        train_df = self._build_features(
            interactions=interactions,
            podcasts=podcasts,
            embedding_scores=embedding_scores,
            collab_scores=collab_scores,
        )

        X = train_df[FEATURE_COLS]
        y = train_df["rating"]

        # --- Train LightGBM ---
        self._model = lgb.LGBMRegressor(**self.params)
        logger.info(f"Training LightGBM ranker on {len(X)} samples...")
        self._model.fit(
            X, y,
            eval_set=[(X, y)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
        )

        # Log feature importances
        importances = pd.Series(
            self._model.feature_importances_, index=FEATURE_COLS
        ).sort_values(ascending=False)
        logger.info(f"Feature importances:\n{importances.to_string()}")

        return self

    # ── Rank ─────────────────────────────────────────────────────────────────

    def rank(
        self,
        user_id: str,
        candidates: pd.DataFrame,
        user_history: Optional[list[str]] = None,
        top_k: int = 10,
        diversity_penalty: float = 0.1,
    ) -> pd.DataFrame:
        """
        Score and rank candidate podcasts for a user.

        Args:
            user_id:          User identifier.
            candidates:       DataFrame of candidates with at minimum:
                              [podcast_id, embedding_score, collab_score]
                              plus any podcast metadata columns.
            user_history:     List of podcast_ids already listened to (excluded).
            top_k:            Number of final recommendations to return.
            diversity_penalty: MMR lambda — higher = more diverse results.

        Returns:
            DataFrame with [podcast_id, title, category, ranker_score],
            sorted by score descending.
        """
        if self._model is None:
            raise RuntimeError("Ranker not trained. Call fit() first.")

        # Exclude already-listened podcasts
        if user_history:
            candidates = candidates[~candidates["podcast_id"].isin(user_history)].copy()

        if candidates.empty:
            return pd.DataFrame(columns=["podcast_id", "ranker_score"])

        # Build features for this user × candidates batch
        features = self._build_inference_features(user_id, candidates)
        X = features[FEATURE_COLS]
        candidates = candidates.copy()
        candidates["ranker_score"] = self._model.predict(X)

        # Normalize scores to [0, 1]
        s_min = candidates["ranker_score"].min()
        s_max = candidates["ranker_score"].max()
        if s_max > s_min:
            candidates["ranker_score"] = (
                (candidates["ranker_score"] - s_min) / (s_max - s_min)
            )

        # Apply MMR (Maximal Marginal Relevance) for diversity
        if diversity_penalty > 0 and "category" in candidates.columns:
            candidates = self._apply_mmr(candidates, top_k, diversity_penalty)
        else:
            candidates = candidates.nlargest(top_k, "ranker_score")

        return candidates.reset_index(drop=True)

    # ── Feature Engineering ──────────────────────────────────────────────────

    def _build_features(
        self,
        interactions: pd.DataFrame,
        podcasts: pd.DataFrame,
        embedding_scores: Optional[pd.DataFrame],
        collab_scores: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Build the training feature matrix."""
        df = interactions[["user_id", "podcast_id", "rating"]].merge(
            podcasts[["podcast_id", "category", "avg_duration_min",
                       "total_episodes", "play_count", "release_year"]],
            on="podcast_id", how="left",
        )

        # Merge pre-computed retrieval scores if available
        if embedding_scores is not None:
            df = df.merge(embedding_scores, on=["user_id", "podcast_id"], how="left")
        else:
            df["embedding_score"] = 0.0

        if collab_scores is not None:
            df = df.merge(collab_scores, on=["user_id", "podcast_id"], how="left")
        else:
            df["collab_score"] = 0.0

        df = self._add_derived_features(df)
        df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)
        return df

    def _build_inference_features(
        self, user_id: str, candidates: pd.DataFrame
    ) -> pd.DataFrame:
        """Build inference features for a single user's candidates."""
        df = candidates.copy()
        df["user_id"] = user_id

        if "avg_duration_min" not in df.columns:
            df["avg_duration_min"] = 60.0
        if "total_episodes" not in df.columns:
            df["total_episodes"] = 100.0
        if "play_count" not in df.columns:
            df["play_count"] = 0.0
        if "release_year" not in df.columns:
            df["release_year"] = 2020.0

        df = self._add_derived_features(df)
        df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)
        return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add normalized and derived features."""
        # Normalize play count
        df["play_count_norm"] = (
            df["play_count"] / self._play_count_max
        ).clip(0, 1)

        # Normalize release year to [0, 1]
        year_range = self._release_year_max - self._release_year_min
        df["release_year_norm"] = (
            (df["release_year"] - self._release_year_min) / max(year_range, 1)
        ).clip(0, 1)

        # Category match: does user like this podcast's category?
        df["category_match"] = df.apply(
            lambda row: self._user_category_prefs.get(
                row.get("user_id", ""), {}
            ).get(row.get("category", ""), 0.0),
            axis=1,
        )

        # Popularity score (log-scaled play count)
        df["popularity_score"] = np.log1p(df["play_count"]) / np.log1p(
            self._play_count_max
        )

        return df

    # ── Diversity (MMR) ───────────────────────────────────────────────────────

    def _apply_mmr(
        self,
        candidates: pd.DataFrame,
        top_k: int,
        lambda_: float,
    ) -> pd.DataFrame:
        """
        Maximal Marginal Relevance re-ranking for diversity.

        Balances relevance (ranker_score) against redundancy (category overlap).
        Higher lambda_ = more diverse; lambda_=0 = pure relevance ranking.

        MMR formula: MMR(d) = λ * relevance(d) - (1-λ) * max_similarity(d, S)
        where S is the set of already-selected items.
        """
        selected = []
        remaining = candidates.copy()

        while len(selected) < top_k and not remaining.empty:
            if not selected:
                # First pick: highest relevance
                best_idx = remaining["ranker_score"].idxmax()
            else:
                selected_categories = [s["category"] for s in selected if "category" in s]
                scores = []
                for _, row in remaining.iterrows():
                    relevance = row["ranker_score"]
                    # Penalize if same category already selected
                    cat_count = selected_categories.count(row.get("category", ""))
                    redundancy = min(cat_count * 0.3, 0.9)
                    mmr_score = lambda_ * relevance - (1 - lambda_) * redundancy
                    scores.append(mmr_score)
                remaining = remaining.copy()
                remaining["mmr_score"] = scores
                best_idx = remaining["mmr_score"].idxmax()

            selected.append(remaining.loc[best_idx].to_dict())
            remaining = remaining.drop(index=best_idx)

        result = pd.DataFrame(selected)
        if "mmr_score" in result.columns:
            result = result.drop(columns=["mmr_score"])
        return result

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate_ndcg(
        self,
        test_interactions: pd.DataFrame,
        podcasts: pd.DataFrame,
        k: int = 10,
    ) -> float:
        """
        Compute NDCG@k on the test set.

        Builds feature vectors for test interactions and computes NDCG
        using the true ratings as relevance scores.
        """
        test_df = test_interactions[["user_id", "podcast_id", "rating"]].merge(
            podcasts[["podcast_id", "category", "avg_duration_min",
                       "total_episodes", "play_count", "release_year"]],
            on="podcast_id", how="left",
        )
        test_df["embedding_score"] = 0.0
        test_df["collab_score"] = 0.0
        test_df = self._add_derived_features(test_df)
        test_df[FEATURE_COLS] = test_df[FEATURE_COLS].fillna(0.0)

        ndcg_scores = []
        for user_id, group in test_df.groupby("user_id"):
            if len(group) < 2:
                continue
            y_true = group["rating"].values.reshape(1, -1)
            y_score = self._model.predict(group[FEATURE_COLS]).reshape(1, -1)
            ndcg_scores.append(ndcg_score(y_true, y_score, k=min(k, len(group))))

        mean_ndcg = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
        logger.info(f"NDCG@{k}: {mean_ndcg:.4f}")
        return mean_ndcg

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"HybridRanker saved to {path}")

    @classmethod
    def load(cls, path: str) -> "HybridRanker":
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"HybridRanker loaded from {path}")
        return model


# ─── Quick test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.data.loader import load_data

    dataset = load_data(use_synthetic=True)
    ranker = HybridRanker(n_estimators=50)

    # For a quick test, we pass None for retrieval scores
    ranker.fit(
        interactions=dataset.train_interactions,
        podcasts=dataset.podcasts,
    )

    ndcg = ranker.evaluate_ndcg(dataset.test_interactions, dataset.podcasts, k=10)
    print(f"NDCG@10: {ndcg:.4f}")

    sample_user = dataset.train_interactions["user_id"].iloc[0]
    candidate_pods = dataset.podcasts.head(50).copy()
    candidate_pods["embedding_score"] = np.random.rand(50)
    candidate_pods["collab_score"] = np.random.rand(50)

    ranked = ranker.rank(sample_user, candidate_pods, top_k=5)
    print(f"\nTop-5 for {sample_user}:")
    print(ranked[["podcast_id", "category", "ranker_score"]].to_string(index=False))
