from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

from src.data.normalize import normalize_rss_jsonl
from src.data.rss_discovery import load_seed_feeds
from src.data.rss_scraper import scrape_feeds


def run_extract(seed_file: str, raw_jsonl: str, metadata_json: str, processed_dir: str) -> dict:
    feeds = load_seed_feeds(seed_file)
    scrape_summary = scrape_feeds(feeds, output_jsonl=raw_jsonl, metadata_path=metadata_json)

    podcasts_df, interactions_df = normalize_rss_jsonl(raw_jsonl)
    output_dir = Path(processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    podcasts_path = output_dir / "podcasts.parquet"
    interactions_path = output_dir / "interactions.parquet"
    podcasts_df.to_parquet(podcasts_path, index=False)
    interactions_df.to_parquet(interactions_path, index=False)

    summary = {
        **scrape_summary,
        "podcasts_rows": int(len(podcasts_df)),
        "interactions_rows": int(len(interactions_df)),
        "podcasts_path": str(podcasts_path),
        "interactions_path": str(interactions_path),
    }
    logger.info(f"RSS extraction summary: {summary}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and normalize RSS feeds.")
    parser.add_argument("--seed-file", default="data/raw/seed_feeds.txt")
    parser.add_argument("--raw-jsonl", default="data/raw/rss_feeds.jsonl")
    parser.add_argument("--metadata-json", default="data/raw/rss_metadata.json")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--summary-path", default="metrics/rss_scrape_summary.json")
    args = parser.parse_args()

    result = run_extract(
        seed_file=args.seed_file,
        raw_jsonl=args.raw_jsonl,
        metadata_json=args.metadata_json,
        processed_dir=args.processed_dir,
    )
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
