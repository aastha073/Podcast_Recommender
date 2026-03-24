from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def normalize_rss_jsonl(input_jsonl: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert raw RSS JSONL into canonical podcast/interactions tables.
    Interactions are implicit and generated from episode recency.
    """
    podcast_rows: list[dict] = []
    interaction_rows: list[dict] = []

    path = Path(input_jsonl)
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame()

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        feed_url = raw.get("feed_url", "")
        feed = raw.get("feed", {})
        podcast_id = _podcast_id(feed_url)
        category = _extract_category(feed)

        podcast_rows.append(
            {
                "podcast_id": podcast_id,
                "title": feed.get("title", "Untitled Podcast"),
                "description": feed.get("subtitle") or feed.get("description") or "",
                "category": category,
                "avg_duration_min": 45,
                "total_episodes": len(raw.get("entries", [])),
                "play_count": max(len(raw.get("entries", [])) * 100, 1),
                "release_year": _extract_release_year(raw.get("entries", [])),
            }
        )

        for idx, entry in enumerate(raw.get("entries", [])):
            guid = entry.get("id") or entry.get("guid") or entry.get("link") or f"{podcast_id}-{idx}"
            timestamp = entry.get("published_parsed") or entry.get("updated") or raw.get("last_scraped_at")
            interaction_rows.append(
                {
                    "user_id": f"rss_user_{idx % 20:03d}",
                    "podcast_id": podcast_id,
                    "episode_guid": str(guid),
                    "rating": 4.0,
                    "timestamp": pd.to_datetime(timestamp, utc=True, errors="coerce"),
                }
            )

    podcasts = pd.DataFrame(podcast_rows).drop_duplicates(subset=["podcast_id"])
    interactions = pd.DataFrame(interaction_rows).drop_duplicates(subset=["podcast_id", "episode_guid"])
    if not interactions.empty:
        interactions["timestamp"] = interactions["timestamp"].fillna(pd.Timestamp.utcnow())
        interactions = interactions.drop(columns=["episode_guid"])
    return podcasts, interactions


def _podcast_id(feed_url: str) -> str:
    digest = hashlib.sha256(feed_url.encode("utf-8")).hexdigest()[:12]
    return f"rss_{digest}"


def _extract_category(feed: dict) -> str:
    tag = feed.get("tags")
    if isinstance(tag, list) and tag:
        first = tag[0]
        if isinstance(first, dict):
            return str(first.get("term", "General"))
        return str(first)
    return "General"


def _extract_release_year(entries: list[dict]) -> int:
    for entry in entries:
        parsed = pd.to_datetime(
            entry.get("published") or entry.get("updated"),
            errors="coerce",
        )
        if pd.notna(parsed):
            return int(parsed.year)
    return pd.Timestamp.utcnow().year
