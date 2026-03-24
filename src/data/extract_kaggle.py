from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd
from loguru import logger


def run_extract(
    dataset: str,
    processed_dir: str,
    raw_dir: str = "data/raw/kaggle",
) -> dict:
    """
    Download a Kaggle dataset and normalize podcast reviews into
    podcasts/interactions parquet files.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    _download_dataset(dataset, raw_path)
    reviews = _load_best_reviews_file(raw_path)
    podcasts_df, interactions_df = _normalize_reviews(reviews)

    podcasts_out = processed_path / "podcasts.parquet"
    interactions_out = processed_path / "interactions.parquet"
    podcasts_df.to_parquet(podcasts_out, index=False)
    interactions_df.to_parquet(interactions_out, index=False)

    summary = {
        "dataset": dataset,
        "rows_reviews": int(len(reviews)),
        "rows_podcasts": int(len(podcasts_df)),
        "rows_interactions": int(len(interactions_df)),
        "podcasts_path": str(podcasts_out),
        "interactions_path": str(interactions_out),
    }
    logger.info(f"Kaggle extraction summary: {summary}")
    return summary


def _download_dataset(dataset: str, output_dir: Path) -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError("kaggle package not installed. Run: pip install kaggle") from exc

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:
        raise RuntimeError(
            "Kaggle authentication failed. Add ~/.kaggle/kaggle.json or set "
            "KAGGLE_USERNAME and KAGGLE_KEY env vars."
        ) from exc

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        api.dataset_download_files(dataset, path=str(tmp_path), unzip=True, quiet=False)
        for f in tmp_path.rglob("*"):
            if f.is_file():
                target = output_dir / f.name
                target.write_bytes(f.read_bytes())


def _load_best_reviews_file(raw_path: Path) -> pd.DataFrame:
    csv_files = list(raw_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_path}")

    scored: list[tuple[int, Path, pd.DataFrame]] = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
        except Exception:
            continue
        cols = {c.lower() for c in df.columns}
        score = 0
        if "rating" in cols or "score" in cols:
            score += 3
        if "review" in cols or "review_text" in cols or "content" in cols:
            score += 3
        if "podcast" in " ".join(cols) or "title" in cols:
            score += 2
        score += min(len(df), 1000) // 100
        scored.append((score, file, df))

    if not scored:
        raise RuntimeError("Unable to parse any CSV files from Kaggle dataset.")
    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][2]
    logger.info(f"Selected reviews file: {scored[0][1].name} (score={scored[0][0]})")
    return best


def _normalize_reviews(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = {c.lower(): c for c in df.columns}

    podcast_col = _pick(cols, ["podcast_title", "podcast_name", "podcast", "title", "show_name"])
    review_col = _pick(cols, ["review_text", "review", "content", "description", "text"])
    rating_col = _pick(cols, ["rating", "score", "stars"])
    user_col = _pick(cols, ["author_id", "user_id", "username", "author", "user"])
    time_col = _pick(cols, ["timestamp", "created_at", "date", "review_date", "published_at"])
    category_col = _pick(cols, ["category", "genre"])

    if podcast_col is None:
        raise ValueError("Could not find a podcast title/name column in Kaggle dataset.")

    work = df.copy()
    work["podcast_title_norm"] = work[podcast_col].fillna("Unknown Podcast").astype(str).str.strip()
    work["podcast_id"] = (
        "kaggle_"
        + work["podcast_title_norm"].str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
    )
    work["review_text_norm"] = (
        work[review_col].fillna("").astype(str) if review_col else ""
    )
    work["rating_norm"] = (
        pd.to_numeric(work[rating_col], errors="coerce").fillna(4.0).clip(0.0, 5.0)
        if rating_col
        else 4.0
    )
    work["user_id_norm"] = (
        work[user_col].fillna("").astype(str).replace("", pd.NA)
        if user_col
        else pd.Series(pd.NA, index=work.index)
    )
    if work["user_id_norm"].isna().any():
        work.loc[work["user_id_norm"].isna(), "user_id_norm"] = [
            f"kaggle_user_{i:06d}" for i in work.index[work["user_id_norm"].isna()]
        ]

    if time_col:
        ts = pd.to_datetime(work[time_col], errors="coerce", utc=True)
    else:
        ts = pd.Series(pd.Timestamp.utcnow(), index=work.index)
    work["timestamp_norm"] = ts.fillna(pd.Timestamp.utcnow())

    podcasts = (
        work.groupby("podcast_id", as_index=False)
        .agg(
            title=("podcast_title_norm", "first"),
            description=("review_text_norm", lambda x: " ".join([v for v in x.head(5) if v]).strip()),
            category=(category_col if category_col else "podcast_title_norm", "first"),
            total_episodes=("podcast_id", "size"),
        )
    )
    podcasts["description"] = podcasts["description"].replace("", "Podcast from Kaggle reviews dataset")
    podcasts["avg_duration_min"] = 45
    podcasts["play_count"] = podcasts["total_episodes"].clip(lower=1) * 100
    podcasts["release_year"] = pd.Timestamp.utcnow().year

    interactions = pd.DataFrame(
        {
            "user_id": work["user_id_norm"].astype(str),
            "podcast_id": work["podcast_id"].astype(str),
            "rating": pd.to_numeric(work["rating_norm"], errors="coerce").fillna(4.0).clip(0, 5),
            "timestamp": work["timestamp_norm"],
        }
    ).drop_duplicates(subset=["user_id", "podcast_id", "timestamp"])

    return podcasts[
        ["podcast_id", "title", "description", "category", "avg_duration_min", "total_episodes", "play_count", "release_year"]
    ], interactions


def _pick(cols: dict[str, str], candidates: list[str]) -> str | None:
    for key in candidates:
        if key in cols:
            return cols[key]
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract podcast reviews data from Kaggle.")
    parser.add_argument("--dataset", default="thoughtvector/podcastreviews")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--raw-dir", default="data/raw/kaggle")
    parser.add_argument("--summary-path", default="metrics/kaggle_scrape_summary.json")
    args = parser.parse_args()

    result = run_extract(dataset=args.dataset, processed_dir=args.processed_dir, raw_dir=args.raw_dir)
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
