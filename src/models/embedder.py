"""
Semantic embedding model for podcast content retrieval.

Uses Sentence-BERT (all-MiniLM-L6-v2) to encode podcast descriptions into
dense 384-dim vectors. Builds a FAISS index for sub-millisecond approximate
nearest-neighbour (ANN) search at serving time.

Why Sentence-BERT?
  - Captures semantic meaning (not just keyword overlap)
  - "machine learning for beginners" ≈ "intro to AI for non-engineers"
  - all-MiniLM-L6-v2: 384-dim, very fast, surprisingly strong quality
  - Swap for all-mpnet-base-v2 (768-dim) if you want higher accuracy

FAISS (Facebook AI Similarity Search):
  - IndexFlatIP: exact cosine similarity search (great up to ~100k items)
  - IndexIVFFlat: faster approximate search for 100k+ items (add training step)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class PodcastEmbedder:
    """
    Encodes podcast descriptions into dense vectors and retrieves
    semantically similar podcasts for a given user query.

    Typical usage:
        embedder = PodcastEmbedder()
        embedder.fit(podcasts_df)
        results = embedder.query("true crime mysteries", top_k=20)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
        max_seq_length: int = 256,
        cache_path: Optional[str] = None,
    ):
        """
        Args:
            model_name:      HuggingFace model name. First run downloads ~90MB.
            batch_size:      Encoding batch size. Increase if you have GPU RAM.
            max_seq_length:  Truncate descriptions to this many tokens.
            cache_path:      If set, save/load embeddings from disk to avoid
                             re-encoding on every run (speeds up iteration).
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.cache_path = Path(cache_path) if cache_path else None

        self._model: Optional[SentenceTransformer] = None
        self._embeddings: Optional[np.ndarray] = None   # (n_podcasts, embedding_dim)
        self._faiss_index: Optional[faiss.IndexFlatIP] = None
        self._podcast_ids: Optional[list[str]] = None
        self._podcast_df: Optional[pd.DataFrame] = None

    # ── Model loading (lazy) ─────────────────────────────────────────────────

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the model on first access."""
        if self._model is None:
            logger.info(f"Loading Sentence-BERT model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            self._model.max_seq_length = self.max_seq_length
        return self._model

    # ── Fitting ──────────────────────────────────────────────────────────────

    def fit(self, podcasts: pd.DataFrame) -> "PodcastEmbedder":
        """
        Encode all podcast descriptions and build the FAISS index.

        Args:
            podcasts: DataFrame with at minimum columns:
                      [podcast_id, title, description, category]

        Returns:
            self (for chaining)
        """
        self._podcast_df = podcasts.reset_index(drop=True)
        self._podcast_ids = podcasts["podcast_id"].tolist()

        # Try loading from cache first
        if self.cache_path and self.cache_path.exists():
            logger.info(f"Loading embeddings from cache: {self.cache_path}")
            self._embeddings = np.load(self.cache_path)
        else:
            self._embeddings = self._encode_podcasts(podcasts)
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(self.cache_path, self._embeddings)
                logger.info(f"Embeddings cached to: {self.cache_path}")

        self._build_faiss_index(self._embeddings)
        logger.info(
            f"Embedder ready. Index contains {len(self._podcast_ids)} podcasts "
            f"({self._embeddings.shape[1]}-dim)."
        )
        return self

    def _encode_podcasts(self, podcasts: pd.DataFrame) -> np.ndarray:
        """
        Encode podcast descriptions into embeddings.

        We concatenate title + category + description for richer context.
        The [SEP] token gives the model a structural hint.
        """
        texts = (
            podcasts["title"].fillna("")
            + " [SEP] "
            + podcasts["category"].fillna("")
            + " [SEP] "
            + podcasts["description"].fillna("")
        ).tolist()

        logger.info(f"Encoding {len(texts)} podcast descriptions...")
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-normalize → cosine sim = dot product
        )
        return embeddings.astype(np.float32)

    def _build_faiss_index(self, embeddings: np.ndarray) -> None:
        """
        Build an in-memory FAISS index for fast ANN search.

        IndexFlatIP = exact inner product (cosine similarity since we L2-
        normalized). For >100k items, swap for IndexIVFFlat with nlist=100.
        """
        dim = embeddings.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dim)
        self._faiss_index.add(embeddings)
        logger.info(f"FAISS index built: {self._faiss_index.ntotal} vectors, dim={dim}.")

    # ── Query ────────────────────────────────────────────────────────────────

    def query(
        self,
        text: str,
        top_k: int = 20,
        exclude_ids: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Find the top-k most semantically similar podcasts to a free-text query.

        Args:
            text:        User's natural-language query or interest description.
            top_k:       Number of results to return.
            exclude_ids: Podcast IDs to exclude (e.g. already-listened podcasts).

        Returns:
            DataFrame with columns [podcast_id, title, category,
            description, embedding_score], sorted by score descending.
        """
        query_vec = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        # Retrieve more candidates to account for exclusions
        fetch_k = min(top_k + len(exclude_ids or []) + 10, len(self._podcast_ids))
        scores, indices = self._faiss_index.search(query_vec, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            pod_id = self._podcast_ids[idx]
            if exclude_ids and pod_id in exclude_ids:
                continue
            row = self._podcast_df.iloc[idx]
            results.append({
                "podcast_id": pod_id,
                "title": row["title"],
                "category": row["category"],
                "description": row["description"],
                "embedding_score": float(score),
            })
            if len(results) >= top_k:
                break

        return pd.DataFrame(results)

    def get_embedding(self, podcast_id: str) -> Optional[np.ndarray]:
        """Return the embedding vector for a specific podcast_id."""
        if podcast_id not in self._podcast_ids:
            return None
        idx = self._podcast_ids.index(podcast_id)
        return self._embeddings[idx]

    def get_all_embeddings(self) -> tuple[list[str], np.ndarray]:
        """Return (podcast_ids, embeddings_matrix) for downstream models."""
        return self._podcast_ids, self._embeddings

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Serialize the embedder (model weights are not saved — re-loaded from HF)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "embeddings.npy", self._embeddings)
        faiss.write_index(self._faiss_index, str(path / "faiss.index"))
        with open(path / "podcast_ids.pkl", "wb") as f:
            pickle.dump(self._podcast_ids, f)
        self._podcast_df.to_parquet(path / "podcasts.parquet")
        logger.info(f"Embedder saved to {path}")

    @classmethod
    def load(cls, path: str, model_name: str = "all-MiniLM-L6-v2") -> "PodcastEmbedder":
        """Load a previously saved embedder."""
        path = Path(path)
        embedder = cls(model_name=model_name)
        embedder._embeddings = np.load(path / "embeddings.npy")
        embedder._faiss_index = faiss.read_index(str(path / "faiss.index"))
        with open(path / "podcast_ids.pkl", "rb") as f:
            embedder._podcast_ids = pickle.load(f)
        embedder._podcast_df = pd.read_parquet(path / "podcasts.parquet")
        logger.info(f"Embedder loaded from {path}")
        return embedder


# ─── Quick test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.data.loader import load_data

    dataset = load_data(use_synthetic=True)

    embedder = PodcastEmbedder(model_name="all-MiniLM-L6-v2", batch_size=64)
    embedder.fit(dataset.podcasts)

    print("\n--- Query: 'artificial intelligence and machine learning' ---")
    results = embedder.query("artificial intelligence and machine learning", top_k=5)
    print(results[["title", "category", "embedding_score"]].to_string(index=False))

    print("\n--- Query: 'true crime murder mystery' ---")
    results = embedder.query("true crime murder mystery", top_k=5)
    print(results[["title", "category", "embedding_score"]].to_string(index=False))
