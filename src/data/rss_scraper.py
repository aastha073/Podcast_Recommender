from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import feedparser
import requests
from loguru import logger


@dataclass
class ScrapeStats:
    feeds_total: int = 0
    feeds_success: int = 0
    feeds_failed: int = 0
    episodes_new: int = 0
    episodes_failed: int = 0


def scrape_feeds(
    feed_urls: list[str],
    output_jsonl: str,
    metadata_path: str,
    timeout_seconds: int = 15,
) -> dict[str, Any]:
    """Fetch RSS feeds and store raw feed/episode payloads as JSONL."""
    output = Path(output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    meta_file = Path(metadata_path)
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    previous_meta = _load_metadata(meta_file)

    stats = ScrapeStats(feeds_total=len(feed_urls))
    now = datetime.now(UTC).isoformat()
    next_meta: dict[str, dict[str, str]] = {}

    with output.open("w", encoding="utf-8") as handle:
        for feed_url in feed_urls:
            headers = _conditional_headers(previous_meta.get(feed_url, {}))
            try:
                response = requests.get(feed_url, timeout=timeout_seconds, headers=headers)
                if response.status_code == 304:
                    stats.feeds_success += 1
                    continue
                response.raise_for_status()
                parsed = feedparser.parse(response.content)
                if parsed.bozo and not parsed.entries:
                    raise ValueError(f"Malformed feed content for {feed_url}")

                entry_count = len(parsed.entries)
                stats.episodes_new += entry_count
                stats.feeds_success += 1

                payload = {
                    "feed_url": feed_url,
                    "last_scraped_at": now,
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                    "feed": _clean(parsed.feed),
                    "entries": [_clean(entry) for entry in parsed.entries],
                }
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

                next_meta[feed_url] = {
                    "etag": response.headers.get("ETag", ""),
                    "last_modified": response.headers.get("Last-Modified", ""),
                }
            except Exception as exc:  # pragma: no cover - network variability
                logger.warning(f"RSS fetch failed for {feed_url}: {exc}")
                stats.feeds_failed += 1
                next_meta[feed_url] = previous_meta.get(feed_url, {})

    summary = {
        "feeds_total": stats.feeds_total,
        "feeds_success": stats.feeds_success,
        "feeds_failed": stats.feeds_failed,
        "episodes_new": stats.episodes_new,
        "episodes_failed": stats.episodes_failed,
        "last_scraped_at": now,
    }
    meta_file.write_text(json.dumps({"feeds": next_meta, "summary": summary}, indent=2), encoding="utf-8")
    return summary


def _conditional_headers(meta: dict[str, str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    etag = meta.get("etag")
    last_modified = meta.get("last_modified")
    if etag:
        headers["If-None-Match"] = etag
    if last_modified:
        headers["If-Modified-Since"] = last_modified
    return headers


def _load_metadata(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw.get("feeds", {})


def _clean(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        data = dict(data)
    safe: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        elif isinstance(value, list):
            safe[key] = [str(v) for v in value[:5]]
        else:
            safe[key] = str(value)
    return safe
