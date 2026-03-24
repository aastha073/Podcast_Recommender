PYTHON ?= python

.PHONY: lint test train-synthetic train-real scrape-smoke

lint:
	ruff check src tests

test:
	pytest tests -v --tb=short

train-synthetic:
	PYTHONPATH=. $(PYTHON) -m src.models.recommender --synthetic --no-mlflow --save-dir models/pipeline

train-real:
	PYTHONPATH=. $(PYTHON) -m src.models.recommender --no-mlflow --save-dir models/pipeline

scrape-smoke:
	PYTHONPATH=. $(PYTHON) -m src.data.extract_rss --seed-file data/raw/seed_feeds.txt --summary-path metrics/rss_scrape_summary.json
