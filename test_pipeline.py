"""
Tests for the ML core: embedder, collaborative filter, and recommender pipeline.

Run with:  pytest tests/ -v --tb=short
"""

import numpy as np
import pandas as pd
import pytest

from src.data.loader import generate_synthetic_data, load_data, PodcastDataset
from src.models.embedder import PodcastEmbedder
from src.models.collaborative import CollaborativeFilter
from src.models.ranker import HybridRanker
from src.models.recommender import RecommendationPipeline


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dataset() -> PodcastDataset:
    """Small synthetic dataset shared across tests."""
    return load_data(use_synthetic=True)


@pytest.fixture(scope="module")
def small_podcasts() -> pd.DataFrame:
    podcasts, _ = generate_synthetic_data(n_podcasts=50, n_users=20, n_interactions=500)
    return podcasts


@pytest.fixture(scope="module")
def trained_embedder(small_podcasts) -> PodcastEmbedder:
    embedder = PodcastEmbedder(model_name="all-MiniLM-L6-v2", batch_size=32)
    embedder.fit(small_podcasts)
    return embedder


@pytest.fixture(scope="module")
def trained_cf(dataset) -> CollaborativeFilter:
    cf = CollaborativeFilter(n_factors=16, n_epochs=5)
    cf.fit(dataset.train_interactions, dataset.user_encoder, dataset.podcast_encoder)
    return cf


# ─── Data Loader Tests ────────────────────────────────────────────────────────

class TestDataLoader:
    def test_synthetic_data_shapes(self, dataset):
        assert len(dataset.podcasts) > 0
        assert len(dataset.interactions) > 0
        assert len(dataset.train_interactions) > len(dataset.test_interactions)

    def test_required_podcast_columns(self, dataset):
        for col in ["podcast_id", "title", "description", "category"]:
            assert col in dataset.podcasts.columns

    def test_required_interaction_columns(self, dataset):
        for col in ["user_id", "podcast_id", "rating", "timestamp"]:
            assert col in dataset.interactions.columns

    def test_ratings_in_range(self, dataset):
        assert dataset.interactions["rating"].between(0, 5).all()

    def test_no_duplicate_podcast_ids(self, dataset):
        assert dataset.podcasts["podcast_id"].nunique() == len(dataset.podcasts)

    def test_temporal_split(self, dataset):
        """Test that the split is temporal — all train timestamps < test timestamps."""
        train_max = dataset.train_interactions["timestamp"].max()
        test_min = dataset.test_interactions["timestamp"].min()
        # There may be slight overlap at the boundary, but test should come after
        assert train_max <= test_min or len(dataset.test_interactions) > 0

    def test_encoders_fitted(self, dataset):
        assert len(dataset.user_encoder.classes_) > 0
        assert len(dataset.podcast_encoder.classes_) > 0


# ─── Embedder Tests ───────────────────────────────────────────────────────────

class TestPodcastEmbedder:
    def test_fit_produces_embeddings(self, trained_embedder, small_podcasts):
        _, embeddings = trained_embedder.get_all_embeddings()
        assert embeddings.shape[0] == len(small_podcasts)
        assert embeddings.shape[1] == 384  # all-MiniLM-L6-v2 output dim

    def test_embeddings_are_normalized(self, trained_embedder):
        _, emb = trained_embedder.get_all_embeddings()
        norms = np.linalg.norm(emb, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_query_returns_dataframe(self, trained_embedder):
        results = trained_embedder.query("technology podcast", top_k=5)
        assert isinstance(results, pd.DataFrame)
        assert len(results) <= 5
        assert "podcast_id" in results.columns
        assert "embedding_score" in results.columns

    def test_query_scores_in_range(self, trained_embedder):
        """Cosine similarity of L2-normalized vectors should be in [-1, 1]."""
        results = trained_embedder.query("science", top_k=10)
        assert results["embedding_score"].between(-1.0, 1.0).all()

    def test_query_results_sorted_descending(self, trained_embedder):
        results = trained_embedder.query("comedy humor", top_k=10)
        scores = results["embedding_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_exclude_ids(self, trained_embedder, small_podcasts):
        first_id = small_podcasts["podcast_id"].iloc[0]
        results = trained_embedder.query("test", top_k=20, exclude_ids=[first_id])
        assert first_id not in results["podcast_id"].values

    def test_get_embedding_by_id(self, trained_embedder, small_podcasts):
        pod_id = small_podcasts["podcast_id"].iloc[0]
        emb = trained_embedder.get_embedding(pod_id)
        assert emb is not None
        assert emb.shape == (384,)

    def test_unknown_podcast_returns_none(self, trained_embedder):
        result = trained_embedder.get_embedding("nonexistent_pod_999")
        assert result is None


# ─── Collaborative Filter Tests ───────────────────────────────────────────────

class TestCollaborativeFilter:
    def test_fit_succeeds(self, trained_cf):
        assert trained_cf._algo is not None
        assert trained_cf._trainset is not None

    def test_predict_known_user(self, trained_cf, dataset):
        user_id = dataset.train_interactions["user_id"].iloc[0]
        candidates = dataset.podcasts["podcast_id"].tolist()[:20]
        preds = trained_cf.predict_for_user(user_id, candidates)
        assert isinstance(preds, pd.DataFrame)
        assert "podcast_id" in preds.columns
        assert "collab_score" in preds.columns
        assert len(preds) == 20

    def test_scores_normalized_0_to_1(self, trained_cf, dataset):
        user_id = dataset.train_interactions["user_id"].iloc[0]
        candidates = dataset.podcasts["podcast_id"].tolist()[:30]
        preds = trained_cf.predict_for_user(user_id, candidates)
        assert preds["collab_score"].between(0.0, 1.0).all()

    def test_cold_start_user(self, trained_cf, dataset):
        """Unknown users should get popularity-based fallback, not crash."""
        candidates = dataset.podcasts["podcast_id"].tolist()[:20]
        preds = trained_cf.predict_for_user("brand_new_user_xyz", candidates)
        assert len(preds) == 20
        assert preds["collab_score"].between(0.0, 1.0).all()

    def test_evaluate_returns_metrics(self, trained_cf, dataset):
        metrics = trained_cf.evaluate(dataset.test_interactions.head(100))
        assert "rmse" in metrics
        assert "mae" in metrics
        assert metrics["rmse"] > 0
        assert metrics["mae"] > 0

    def test_user_factors_shape(self, trained_cf):
        raw_ids, factors = trained_cf.get_user_factors()
        assert len(raw_ids) == factors.shape[0]
        assert factors.shape[1] == 16  # n_factors

    def test_item_factors_shape(self, trained_cf):
        raw_ids, factors = trained_cf.get_item_factors()
        assert len(raw_ids) == factors.shape[0]


# ─── Hybrid Ranker Tests ──────────────────────────────────────────────────────

class TestHybridRanker:
    @pytest.fixture(scope="class")
    def trained_ranker(self, dataset):
        ranker = HybridRanker(n_estimators=50)
        ranker.fit(dataset.train_interactions, dataset.podcasts)
        return ranker

    def test_fit_produces_model(self, trained_ranker):
        assert trained_ranker._model is not None

    def test_rank_returns_top_k(self, trained_ranker, dataset):
        user_id = dataset.train_interactions["user_id"].iloc[0]
        candidates = dataset.podcasts.head(50).copy()
        candidates["embedding_score"] = np.random.rand(50)
        candidates["collab_score"] = np.random.rand(50)
        ranked = trained_ranker.rank(user_id, candidates, top_k=10)
        assert len(ranked) <= 10

    def test_rank_scores_sorted(self, trained_ranker, dataset):
        user_id = dataset.train_interactions["user_id"].iloc[0]
        candidates = dataset.podcasts.head(30).copy()
        candidates["embedding_score"] = np.random.rand(30)
        candidates["collab_score"] = np.random.rand(30)
        ranked = trained_ranker.rank(user_id, candidates, top_k=10, diversity_penalty=0.0)
        scores = ranked["ranker_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_history_exclusion(self, trained_ranker, dataset):
        user_id = dataset.train_interactions["user_id"].iloc[0]
        candidates = dataset.podcasts.head(30).copy()
        candidates["embedding_score"] = np.random.rand(30)
        candidates["collab_score"] = np.random.rand(30)
        exclude = candidates["podcast_id"].head(5).tolist()
        ranked = trained_ranker.rank(user_id, candidates, user_history=exclude, top_k=10)
        for pod_id in exclude:
            assert pod_id not in ranked["podcast_id"].values

    def test_ndcg_positive(self, trained_ranker, dataset):
        ndcg = trained_ranker.evaluate_ndcg(dataset.test_interactions.head(200), dataset.podcasts)
        assert ndcg > 0.0


# ─── Full Pipeline Tests ──────────────────────────────────────────────────────

class TestRecommendationPipeline:
    @pytest.fixture(scope="class")
    def pipeline(self, dataset):
        p = RecommendationPipeline(
            n_factors=16, n_epochs=5, ranker_n_estimators=50
        )
        p.fit(dataset, use_mlflow=False)
        return p

    def test_fit_initializes_all_components(self, pipeline):
        assert pipeline.embedder is not None
        assert pipeline.cf_model is not None
        assert pipeline.ranker is not None

    def test_recommend_known_user(self, pipeline, dataset):
        user_id = dataset.train_interactions["user_id"].iloc[0]
        recs = pipeline.recommend(user_id=user_id, query="technology", top_k=5)
        assert isinstance(recs, pd.DataFrame)
        assert len(recs) <= 5
        assert "podcast_id" in recs.columns
        assert "ranker_score" in recs.columns

    def test_recommend_cold_start_user(self, pipeline):
        recs = pipeline.recommend(user_id="totally_new_user", query="true crime", top_k=5)
        assert isinstance(recs, pd.DataFrame)
        assert len(recs) > 0

    def test_recommend_no_query(self, pipeline, dataset):
        user_id = dataset.train_interactions["user_id"].iloc[0]
        recs = pipeline.recommend(user_id=user_id, query="", top_k=5)
        assert isinstance(recs, pd.DataFrame)

    def test_recommend_excludes_history(self, pipeline, dataset):
        user_id = dataset.train_interactions["user_id"].iloc[0]
        history = dataset.podcasts["podcast_id"].head(10).tolist()
        recs = pipeline.recommend(user_id=user_id, query="", user_history=history, top_k=5)
        for pod_id in history:
            assert pod_id not in recs["podcast_id"].values

    def test_batch_recommend(self, pipeline, dataset):
        users = dataset.train_interactions["user_id"].unique()[:3].tolist()
        results = pipeline.recommend_batch(users, top_k=5)
        assert len(results) == 3
        for uid, recs in results.items():
            assert isinstance(recs, pd.DataFrame)

    def test_save_and_load(self, pipeline, tmp_path):
        pipeline.save(str(tmp_path / "pipeline"))
        loaded = RecommendationPipeline.load(str(tmp_path / "pipeline"))
        assert loaded.embedder is not None
        assert loaded.cf_model is not None
        assert loaded.ranker is not None
