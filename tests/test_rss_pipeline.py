from __future__ import annotations

import json

import pandas as pd

from src.data.normalize import normalize_rss_jsonl


def test_normalize_rss_jsonl(tmp_path):
    jsonl_path = tmp_path / "rss_feeds.jsonl"
    row = {
        "feed_url": "https://example.com/feed.xml",
        "last_scraped_at": "2026-03-24T00:00:00Z",
        "feed": {"title": "Example Feed", "description": "Sample podcast feed"},
        "entries": [
            {"id": "ep-1", "published": "2026-01-01T00:00:00Z"},
            {"id": "ep-2", "published": "2026-01-02T00:00:00Z"},
        ],
    }
    jsonl_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    podcasts, interactions = normalize_rss_jsonl(str(jsonl_path))
    assert isinstance(podcasts, pd.DataFrame)
    assert isinstance(interactions, pd.DataFrame)
    assert len(podcasts) == 1
    assert len(interactions) == 2
    assert {"podcast_id", "title", "description", "category"}.issubset(podcasts.columns)
    assert {"user_id", "podcast_id", "rating", "timestamp"}.issubset(interactions.columns)
