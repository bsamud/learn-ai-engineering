"""
Embedding Utilities for AI Engineering
======================================

Embedding models, similarity search, and vector operations.
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import Optional


# =============================================================================
# Embedding Model
# =============================================================================


@dataclass
class EmbeddingResult:
    """Result from embedding operation."""

    text: str
    embedding: list[float]
    model: str
    dimensions: int


class EmbeddingModel:
    """
    Unified interface for embedding models.

    Supports OpenAI and sentence-transformers.

    Example
    -------
    >>> model = EmbeddingModel(provider="openai")
    >>> embedding = model.embed("Hello, world!")
    >>> print(len(embedding))
    1536
    """

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
    ):
        self.provider = provider.lower()

        # Set default model
        if model is None:
            self.model = {
                "openai": "text-embedding-3-small",
                "sentence-transformers": "all-MiniLM-L6-v2",
            }.get(self.provider, "text-embedding-3-small")
        else:
            self.model = model

        self._init_model()

    def _init_model(self) -> None:
        """Initialize the embedding model."""
        if self.provider == "openai":
            from openai import OpenAI

            self.client = OpenAI()

        elif self.provider == "sentence-transformers":
            from sentence_transformers import SentenceTransformer

            self.client = SentenceTransformer(self.model)

    def embed(self, text: str) -> list[float]:
        """
        Embed a single text.

        Parameters
        ----------
        text : str
            Text to embed

        Returns
        -------
        list[float]
            Embedding vector
        """
        if self.provider == "openai":
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
            )
            return response.data[0].embedding

        else:  # sentence-transformers
            embedding = self.client.encode(text)
            return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts.

        Parameters
        ----------
        texts : list[str]
            Texts to embed

        Returns
        -------
        list[list[float]]
            List of embedding vectors
        """
        if self.provider == "openai":
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
            )
            return [item.embedding for item in response.data]

        else:  # sentence-transformers
            embeddings = self.client.encode(texts)
            return [e.tolist() for e in embeddings]

    def get_dimensions(self) -> int:
        """Get embedding dimensions for this model."""
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
        }
        return dimensions.get(self.model, 1536)


# =============================================================================
# Similarity Functions
# =============================================================================


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Parameters
    ----------
    vec1, vec2 : list[float]
        Vectors to compare

    Returns
    -------
    float
        Similarity score (-1 to 1)
    """
    a = np.array(vec1)
    b = np.array(vec2)

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def euclidean_distance(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate Euclidean distance between two vectors.

    Parameters
    ----------
    vec1, vec2 : list[float]
        Vectors to compare

    Returns
    -------
    float
        Distance (lower = more similar)
    """
    a = np.array(vec1)
    b = np.array(vec2)
    return np.linalg.norm(a - b)


def similarity_search(
    query_embedding: list[float],
    embeddings: list[list[float]],
    texts: list[str],
    top_k: int = 5,
    metric: str = "cosine",
) -> list[tuple[str, float]]:
    """
    Find most similar texts to a query.

    Parameters
    ----------
    query_embedding : list[float]
        Query embedding vector
    embeddings : list[list[float]]
        List of document embeddings
    texts : list[str]
        Corresponding texts
    top_k : int
        Number of results to return
    metric : str
        Similarity metric: "cosine" or "euclidean"

    Returns
    -------
    list[tuple[str, float]]
        List of (text, score) tuples, sorted by relevance
    """
    scores = []

    for i, emb in enumerate(embeddings):
        if metric == "cosine":
            score = cosine_similarity(query_embedding, emb)
        else:  # euclidean
            score = -euclidean_distance(query_embedding, emb)  # Negate for sorting

        scores.append((texts[i], score, i))

    # Sort by score (descending)
    scores.sort(key=lambda x: x[1], reverse=True)

    return [(text, score) for text, score, _ in scores[:top_k]]


# =============================================================================
# Simple Vector Store
# =============================================================================


class SimpleVectorStore:
    """
    Simple in-memory vector store.

    For production, use ChromaDB, Pinecone, or similar.

    Example
    -------
    >>> store = SimpleVectorStore(embedding_model)
    >>> store.add(["doc1", "doc2", "doc3"])
    >>> results = store.search("query text", k=2)
    """

    def __init__(self, embedding_model: Optional[EmbeddingModel] = None):
        self.embedding_model = embedding_model or EmbeddingModel()
        self.texts: list[str] = []
        self.embeddings: list[list[float]] = []
        self.metadata: list[dict] = []

    def add(
        self,
        texts: list[str],
        metadata: Optional[list[dict]] = None,
    ) -> None:
        """
        Add texts to the store.

        Parameters
        ----------
        texts : list[str]
            Texts to add
        metadata : list[dict], optional
            Metadata for each text
        """
        embeddings = self.embedding_model.embed_batch(texts)

        self.texts.extend(texts)
        self.embeddings.extend(embeddings)

        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(texts))

        print(f"Added {len(texts)} documents. Total: {len(self.texts)}")

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> list[dict]:
        """
        Search for similar documents.

        Parameters
        ----------
        query : str
            Search query
        k : int
            Number of results

        Returns
        -------
        list[dict]
            Search results with text, score, and metadata
        """
        query_embedding = self.embedding_model.embed(query)

        results = []
        for i, emb in enumerate(self.embeddings):
            score = cosine_similarity(query_embedding, emb)
            results.append({
                "text": self.texts[i],
                "score": score,
                "metadata": self.metadata[i],
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    def __len__(self) -> int:
        return len(self.texts)
