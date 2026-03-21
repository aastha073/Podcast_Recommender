# Podcast Recommendation System + MLOps Pipeline

An end-to-end ML system that recommends podcasts using semantic embeddings,
collaborative filtering, and a hybrid LightGBM ranker — wrapped in a full
MLOps pipeline (MLflow, DVC, FastAPI, Docker, Evidently, GitHub Actions).

## Architecture
```
Raw Data → Sentence-BERT Embeddings → Collaborative Filter (SVD)
                                              ↓
                              Hybrid LightGBM Ranker → FastAPI → Streamlit UI
                                              ↓
                              MLflow Tracking + Model Registry
                                              ↓
                              Evidently Monitoring + GitHub Actions CI/CD
```

## Project Structure
```
podcast-recsys/
├── configs/              # Hydra config files
│   └── config.yaml
├── data/
│   ├── raw/              # Raw podcast metadata, user logs (DVC tracked)
│   └── processed/        # Embeddings, feature matrices
├── src/
│   ├── data/
│   │   ├── loader.py         # Data loading & validation
│   │   └── preprocess.py     # Cleaning & feature engineering
│   ├── features/
│   │   └── feature_store.py  # Feast feature store interface
│   ├── models/
│   │   ├── embedder.py       # Sentence-BERT embedding model
│   │   ├── collaborative.py  # SVD collaborative filtering
│   │   ├── ranker.py         # LightGBM hybrid ranker
│   │   └── recommender.py    # Unified recommendation pipeline
├── tests/
│   ├── test_embedder.py
│   ├── test_collaborative.py
│   └── test_recommender.py
├── notebooks/
│   └── 01_eda.ipynb
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── dvc.yaml              # DVC pipeline stages
└── .github/workflows/    # CI/CD pipelines
```

## Quickstart
```bash
# 1. Clone and install
git clone <repo>
cd podcast-recsys
pip install -r requirements.txt

# 2. Pull data (DVC)
dvc pull

# 3. Run the full training pipeline
python -m src.models.recommender --config configs/config.yaml

# 4. Launch MLflow UI
mlflow ui --port 5000

# 5. Serve the API
uvicorn src.api.main:app --reload
```

## MLOps Components
| Component | Tool | Purpose |
|---|---|---|
| Experiment tracking | MLflow | Log params, metrics, artifacts |
| Data versioning | DVC | Reproducible data pipelines |
| Feature store | Feast | Online (Redis) + offline (Parquet) |
| Model serving | FastAPI + Docker | Low-latency REST API |
| Monitoring | Evidently | Drift detection, perf decay |
| CI/CD | GitHub Actions | Test → train → deploy |
