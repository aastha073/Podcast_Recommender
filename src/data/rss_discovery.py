from __future__ import annotations

from pathlib import Path


def load_seed_feeds(seed_file: str) -> list[str]:
    """Load feed URLs from a newline-delimited text file."""
    path = Path(seed_file)
    if not path.exists():
        raise FileNotFoundError(f"Seed feed file not found: {seed_file}")

    feeds: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        feeds.append(value)
    return feeds
