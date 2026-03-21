"""
Data loader for the podcast recommendation system.

Handles:
- Loading raw podcast metadata and user interaction logs
- Data validation with Great Expectations
- Synthetic data generation for local development/testing
- Train/test splitting with temporal awareness
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ─── Data Schemas ────────────────────────────────────────────────────────────

@dataclass
class PodcastDataset:
    """Container for all data splits used by the recommendation pipeline."""
    podcasts: pd.DataFrame          # Podcast metadata (id, title, description, category, ...)
    interactions: pd.DataFrame      # User-podcast interactions (user_id, podcast_id, rating, timestamp)
    train_interactions: pd.DataFrame
    test_interactions: pd.DataFrame
    user_encoder: LabelEncoder      # Maps user_id strings → int indices
    podcast_encoder: LabelEncoder   # Maps podcast_id strings → int indices


# ─── Synthetic Data Generator ────────────────────────────────────────────────

def generate_synthetic_data(
    n_podcasts: int = 500,
    n_users: int = 1000,
    n_interactions: int = 20_000,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic podcast + interaction data for local development.

    The synthetic data mimics realistic distributions:
    - Podcast popularity follows a power law (a few blockbuster shows)
    - User activity follows a long tail (most users listen to few podcasts)
    - Ratings cluster around categories (users prefer certain genres)

    Args:
        n_podcasts:     Number of unique podcasts to generate.
        n_users:        Number of unique users to generate.
        n_interactions: Total user-podcast interaction events.
        random_seed:    For reproducibility.

    Returns:
        (podcasts_df, interactions_df) tuple.
    """
    rng = np.random.default_rng(random_seed)

    categories = [
        "True Crime", "Technology", "Business", "Comedy", "Science",
        "History", "Health", "Society & Culture", "Sports", "Politics",
    ]
    category_descriptions = {
        "True Crime": "Investigative stories about real crimes and mysteries",
        "Technology": "Deep dives into software, AI, and the future of tech",
        "Business": "Entrepreneurship, finance, and career growth strategies",
        "Comedy": "Stand-up specials, improv, and humorous storytelling",
        "Science": "Exploring the frontiers of physics, biology, and space",
        "History": "Revisiting pivotal moments and forgotten figures in history",
        "Health": "Evidence-based advice on mental and physical wellness",
        "Society & Culture": "Conversations about identity, community, and modern life",
        "Sports": "Analysis, interviews, and storytelling from the sports world",
        "Politics": "Balanced coverage of policy, elections, and current affairs",
    }

    # --- Podcasts ---
    podcast_categories = rng.choice(categories, size=n_podcasts)
    podcasts = pd.DataFrame({
        "podcast_id": [f"pod_{i:04d}" for i in range(n_podcasts)],
        "title": [
            f"{cat} Podcast #{i}" for i, cat in enumerate(podcast_categories)
        ],
        "description": [
            f"{category_descriptions[cat]}. Episode {rng.integers(1, 500)}."
            for cat in podcast_categories
        ],
        "category": podcast_categories,
        "avg_duration_min": rng.integers(20, 120, size=n_podcasts),
        "total_episodes": rng.integers(10, 1000, size=n_podcasts),
        # Power-law popularity: most podcasts have few plays, a few have many
        "play_count": rng.power(0.3, size=n_podcasts) * 1_000_000,
        "release_year": rng.integers(2015, 2024, size=n_podcasts),
    })

    # --- User-Podcast Interactions ---
    # Users prefer categories — sample user category affinity
    user_category_affinity = rng.dirichlet(
        np.ones(len(categories)) * 0.5, size=n_users
    )  # (n_users, n_categories) — rows sum to 1

    # Map podcast index → category index
    cat_index = {cat: i for i, cat in enumerate(categories)}
    podcast_cat_idx = np.array([cat_index[c] for c in podcast_categories])

    # Sample interactions: each user has a random number of interactions
    user_interaction_counts = rng.negative_binomial(5, 0.3, size=n_users)
    user_interaction_counts = np.clip(user_interaction_counts, 1, 200)

    records = []
    for user_idx in range(n_users):
        n = user_interaction_counts[user_idx]
        affinity = user_category_affinity[user_idx]

        # Sample podcasts weighted by user's category affinity
        podcast_weights = affinity[podcast_cat_idx]
        podcast_weights /= podcast_weights.sum()
        chosen = rng.choice(n_podcasts, size=min(n, n_podcasts), replace=False, p=podcast_weights)

        for pod_idx in chosen:
            # Rating: noisy signal biased by category affinity
            base = affinity[podcast_cat_idx[pod_idx]] * 5
            rating = float(np.clip(rng.normal(base, 0.8), 0.5, 5.0))
            timestamp = pd.Timestamp("2022-01-01") + pd.Timedelta(
                days=int(rng.integers(0, 730))
            )
            records.append({
                "user_id": f"user_{user_idx:04d}",
                "podcast_id": podcasts.iloc[pod_idx]["podcast_id"],
                "rating": round(rating, 1),
                "timestamp": timestamp,
            })

    interactions = pd.DataFrame(records)
    # Downsample to requested count if necessary
    if len(interactions) > n_interactions:
        interactions = interactions.sample(n=n_interactions, random_state=random_seed)

    interactions = interactions.reset_index(drop=True)
    logger.info(
        f"Generated {len(podcasts)} podcasts, {n_users} users, "
        f"{len(interactions)} interactions."
    )
    return podcasts, interactions


# ─── Main Loader ─────────────────────────────────────────────────────────────

def load_data(
    podcasts_path: Optional[str] = None,
    interactions_path: Optional[str] = None,
    test_size: float = 0.2,
    min_interactions: int = 5,
    random_seed: int = 42,
    use_synthetic: bool = False,
) -> PodcastDataset:
    """
    Load, validate, and split podcast recommendation data.

    If paths are not provided (or use_synthetic=True), generates synthetic
    data for local development. In production, pass paths to real CSVs.

    Expected CSV schemas:
        podcasts.csv:     podcast_id, title, description, category, ...
        interactions.csv: user_id, podcast_id, rating, timestamp

    Args:
        podcasts_path:    Path to podcast metadata CSV.
        interactions_path: Path to user interactions CSV.
        test_size:        Fraction of interactions for the test set.
        min_interactions: Drop users with fewer than this many interactions.
        random_seed:      For reproducibility.
        use_synthetic:    Force synthetic data even if paths are provided.

    Returns:
        PodcastDataset with train/test splits and fitted encoders.
    """
    if use_synthetic or (podcasts_path is None or interactions_path is None):
        logger.info("Using synthetic data for development.")
        podcasts, interactions = generate_synthetic_data(random_seed=random_seed)
    else:
        logger.info(f"Loading podcasts from {podcasts_path}")
        podcasts = pd.read_csv(podcasts_path)
        logger.info(f"Loading interactions from {interactions_path}")
        interactions = pd.read_csv(interactions_path, parse_dates=["timestamp"])

    # --- Validation ---
    _validate_podcasts(podcasts)
    _validate_interactions(interactions)

    # --- Filter sparse users ---
    user_counts = interactions["user_id"].value_counts()
    active_users = user_counts[user_counts >= min_interactions].index
    interactions = interactions[interactions["user_id"].isin(active_users)].copy()
    logger.info(
        f"After filtering (min_interactions={min_interactions}): "
        f"{interactions['user_id'].nunique()} users, "
        f"{len(interactions)} interactions."
    )

    # --- Encode IDs to integers ---
    user_encoder = LabelEncoder()
    podcast_encoder = LabelEncoder()
    interactions["user_idx"] = user_encoder.fit_transform(interactions["user_id"])
    interactions["podcast_idx"] = podcast_encoder.fit_transform(interactions["podcast_id"])

    # --- Temporal train/test split (realistic: test on most recent data) ---
    interactions = interactions.sort_values("timestamp")
    split_idx = int(len(interactions) * (1 - test_size))
    train_interactions = interactions.iloc[:split_idx].copy()
    test_interactions = interactions.iloc[split_idx:].copy()

    logger.info(
        f"Train: {len(train_interactions)} interactions | "
        f"Test: {len(test_interactions)} interactions"
    )

    return PodcastDataset(
        podcasts=podcasts,
        interactions=interactions,
        train_interactions=train_interactions,
        test_interactions=test_interactions,
        user_encoder=user_encoder,
        podcast_encoder=podcast_encoder,
    )


# ─── Validation ──────────────────────────────────────────────────────────────

def _validate_podcasts(df: pd.DataFrame) -> None:
    required = {"podcast_id", "title", "description", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"podcasts DataFrame missing columns: {missing}")
    assert df["podcast_id"].nunique() == len(df), "Duplicate podcast_ids detected!"
    assert df["description"].notna().all(), "Some podcasts have null descriptions!"
    logger.info(f"Podcasts validated: {len(df)} rows, {df['category'].nunique()} categories.")


def _validate_interactions(df: pd.DataFrame) -> None:
    required = {"user_id", "podcast_id", "rating", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"interactions DataFrame missing columns: {missing}")
    assert df["rating"].between(0, 5).all(), "Ratings must be in [0, 5]."
    logger.info(
        f"Interactions validated: {len(df)} rows, "
        f"{df['user_id'].nunique()} users, {df['podcast_id'].nunique()} podcasts."
    )


# ─── Quick sanity check ──────────────────────────────────────────────────────

if __name__ == "__main__":
    dataset = load_data(use_synthetic=True)
    print(dataset.podcasts.head())
    print(dataset.train_interactions.head())
    print(f"\nTrain size: {len(dataset.train_interactions)}")
    print(f"Test size:  {len(dataset.test_interactions)}")
