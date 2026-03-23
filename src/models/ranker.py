"""
Hybrid ranker using sklearn GradientBoostingRegressor.
Replaced LightGBM which segfaults on macOS Apple Silicon + Python 3.13.
Same interface, pure Python, no native extensions beyond what sklearn uses.
"""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import ndcg_score

FEATURE_COLS = [
    "embedding_score", "collab_score",
    "avg_duration_min", "total_episodes",
    "play_count_norm", "release_year_norm",
    "category_match", "popularity_score",
]

class HybridRanker:
    def __init__(self, n_estimators=100, learning_rate=0.05,
                 num_leaves=63, random_seed=42, **kwargs):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self._model: Optional[GradientBoostingRegressor] = None
        self._user_category_prefs: dict = {}
        self._play_count_max = 1.0
        self._release_year_min = 2010.0
        self._release_year_max = 2024.0

    def fit(self, interactions, podcasts, embedding_scores=None, collab_scores=None):
        logger.info("Building ranker training features...")
        merged = interactions.merge(podcasts[["podcast_id","category"]], on="podcast_id", how="left")
        for user_id, grp in merged.groupby("user_id"):
            self._user_category_prefs[user_id] = grp["category"].value_counts(normalize=True).to_dict()
        self._play_count_max = podcasts["play_count"].max()
        self._release_year_min = podcasts["release_year"].min()
        self._release_year_max = podcasts["release_year"].max()
        train_df = interactions[["user_id","podcast_id","rating"]].merge(
            podcasts[["podcast_id","category","avg_duration_min","total_episodes","play_count","release_year"]],
            on="podcast_id", how="left")
        train_df["embedding_score"] = 0.0
        train_df["collab_score"] = 0.0
        train_df = self._add_derived_features(train_df)
        train_df[FEATURE_COLS] = train_df[FEATURE_COLS].fillna(0.0)
        X, y = train_df[FEATURE_COLS], train_df["rating"]
        self._model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=5, random_state=42, verbose=0)
        logger.info(f"Training GradientBoostingRegressor on {len(X)} samples...")
        self._model.fit(X, y)
        logger.info("Ranker training complete.")
        importances = pd.Series(self._model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
        logger.info(f"Feature importances:\n{importances.to_string()}")
        return self

    def rank(self, user_id, candidates, user_history=None, top_k=10, diversity_penalty=0.1):
        if self._model is None:
            raise RuntimeError("Ranker not trained.")
        if user_history:
            candidates = candidates[~candidates["podcast_id"].isin(user_history)].copy()
        if candidates.empty:
            return pd.DataFrame(columns=["podcast_id","ranker_score"])
        df = candidates.copy()
        df["user_id"] = user_id
        for col in ["avg_duration_min","total_episodes","play_count","release_year"]:
            if col not in df.columns:
                df[col] = 60.0 if col == "avg_duration_min" else 100.0
        df = self._add_derived_features(df)
        df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)
        candidates = candidates.copy()
        candidates["ranker_score"] = self._model.predict(df[FEATURE_COLS])
        lo, hi = candidates["ranker_score"].min(), candidates["ranker_score"].max()
        if hi > lo:
            candidates["ranker_score"] = (candidates["ranker_score"] - lo) / (hi - lo)
        if diversity_penalty > 0 and "category" in candidates.columns:
            candidates = self._apply_mmr(candidates, top_k, diversity_penalty)
        else:
            candidates = candidates.nlargest(top_k, "ranker_score")
        return candidates.reset_index(drop=True)

    def evaluate_ndcg(self, test_interactions, podcasts, k=10):
        test_df = test_interactions[["user_id","podcast_id","rating"]].merge(
            podcasts[["podcast_id","category","avg_duration_min","total_episodes","play_count","release_year"]],
            on="podcast_id", how="left")
        test_df["embedding_score"] = 0.0
        test_df["collab_score"] = 0.0
        test_df = self._add_derived_features(test_df)
        test_df[FEATURE_COLS] = test_df[FEATURE_COLS].fillna(0.0)
        scores = []
        for _, group in test_df.groupby("user_id"):
            if len(group) < 2: continue
            y_true = group["rating"].values.reshape(1,-1)
            y_score = self._model.predict(group[FEATURE_COLS]).reshape(1,-1)
            scores.append(ndcg_score(y_true, y_score, k=min(k, len(group))))
        mean_ndcg = float(np.mean(scores)) if scores else 0.0
        logger.info(f"NDCG@{k}: {mean_ndcg:.4f}")
        return mean_ndcg

    def _add_derived_features(self, df):
        df = df.copy()
        df["play_count_norm"] = (df.get("play_count", 0) / self._play_count_max).clip(0, 1)
        yr = self._release_year_max - self._release_year_min
        df["release_year_norm"] = ((df.get("release_year", 2020) - self._release_year_min) / max(yr, 1)).clip(0, 1)
        df["category_match"] = df.apply(
            lambda r: self._user_category_prefs.get(r.get("user_id",""), {}).get(r.get("category",""), 0.0), axis=1)
        df["popularity_score"] = np.log1p(df.get("play_count", 0)) / np.log1p(self._play_count_max)
        return df

    def _apply_mmr(self, candidates, top_k, lambda_):
        selected, remaining = [], candidates.copy()
        while len(selected) < top_k and not remaining.empty:
            if not selected:
                best_idx = remaining["ranker_score"].idxmax()
            else:
                sel_cats = [s.get("category","") for s in selected]
                scores = []
                for _, row in remaining.iterrows():
                    redundancy = min(sel_cats.count(row.get("category","")) * 0.3, 0.9)
                    scores.append(lambda_ * row["ranker_score"] - (1-lambda_) * redundancy)
                remaining = remaining.copy()
                remaining["mmr_score"] = scores
                best_idx = remaining["mmr_score"].idxmax()
            selected.append(remaining.loc[best_idx].to_dict())
            remaining = remaining.drop(index=best_idx)
        result = pd.DataFrame(selected)
        if "mmr_score" in result.columns:
            result = result.drop(columns=["mmr_score"])
        return result

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
