"""
retriever.py — ChromaDB-backed vector store with cross-encoder re-ranking.

Architecture
------------
                     ┌──────────────┐
  query ────────────►│ EmbeddingModel│──► query vector
                     └──────────────┘
                            │
                     ┌──────▼──────────────┐
                     │  ChromaDB Collection │  top_k ANN search
                     └──────┬──────────────┘
                            │  top_k candidates
                     ┌──────▼──────────────┐
                     │   CrossEncoder       │  re-rank (optional)
                     └──────┬──────────────┘
                            │  final_k results (filtered by threshold)
                     ┌──────▼──────────────┐
                     │  RetrievalResult[]   │  returned to generator
                     └─────────────────────┘

Design choices
--------------
* Each (embedding_model × chunking_strategy) combination gets its own
  ChromaDB collection so indexes are never mixed or overwritten.
* We store chunk text + all metadata in ChromaDB so the retriever is
  self-contained and doesn't need to re-read JSON files at query time.
* The cross-encoder re-ranker uses a different (asymmetric) model to the
  bi-encoder, which consistently improves precision in IR benchmarks.
* Similarity threshold filtering prevents the generator from receiving
  irrelevant context, which is the primary cause of hallucinations.

Usage
-----
    from retriever import Retriever
    r = Retriever(embedding_model_slug="bge-base-en-v1.5", chunking_strategy="section")
    r.index(chunks)
    results = r.retrieve("What is the attention mechanism?")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chromadb
import numpy as np
from chromadb.config import Settings
from sentence_transformers import CrossEncoder
from tqdm import tqdm

from chunking import Chunk
from embeddings import EmbeddingModel, get_embedding_model
from utils import get_config, get_logger

logger = get_logger(__name__)
cfg = get_config()

# ── Data Model ────────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """A single retrieved passage with its relevance scores."""

    chunk:           Chunk
    vector_score:    float    # cosine similarity from ChromaDB (0–1)
    reranker_score:  float    # cross-encoder logit (higher = more relevant)
    rank:            int      # final rank after re-ranking

    def above_threshold(self) -> bool:
        threshold = cfg["retrieval"]["similarity_threshold"]
        return self.vector_score >= threshold


# ── Retriever ─────────────────────────────────────────────────────────────────

class Retriever:
    """
    Manages a ChromaDB collection and retrieves relevant chunks for a query.

    Parameters
    ----------
    embedding_model_slug : model key from config.yaml (e.g. "bge-base-en-v1.5")
    chunking_strategy    : "fixed" | "recursive" | "section"
    chroma_dir           : path to ChromaDB persistence directory
    """

    def __init__(
        self,
        embedding_model_slug: str | None = None,
        chunking_strategy:    str | None = None,
        chroma_dir:           Path | None = None,
    ) -> None:
        self.embed_slug = embedding_model_slug or cfg["embeddings"]["default_model"]
        self.strategy   = chunking_strategy    or cfg["chunking"]["default_strategy"]

        root = Path(__file__).parent
        self.chroma_dir = chroma_dir or (root / cfg["paths"]["chroma_dir"])
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

        # Collection name encodes the experiment configuration
        slug_safe   = self.embed_slug.replace("-", "_").replace(".", "_")
        strat_safe  = self.strategy
        prefix      = cfg["chroma"]["collection_prefix"]
        self.collection_name = f"{prefix}_{slug_safe}_{strat_safe}"

        # ChromaDB client
        self._client = chromadb.PersistentClient(
            path=str(self.chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        distance = cfg["chroma"]["distance_metric"]
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": distance},
        )
        logger.info(
            "ChromaDB collection '%s' ready (%d documents).",
            self.collection_name,
            self._collection.count(),
        )

        # Embedding model
        self._embed_model: EmbeddingModel = get_embedding_model(self.embed_slug)

        # Cross-encoder for re-ranking
        self._reranker: CrossEncoder | None = None
        if cfg["retrieval"]["use_reranker"]:
            reranker_name = cfg["retrieval"]["reranker_model"]
            logger.info("Loading re-ranker '%s' …", reranker_name)
            self._reranker = CrossEncoder(reranker_name)

    # ── Indexing ──────────────────────────────────────────────────────────

    def index(self, chunks: list[Chunk], batch_size: int = 128) -> None:
        """
        Embed and upsert chunks into ChromaDB.

        Existing chunks with the same chunk_id are overwritten (upsert).

        Parameters
        ----------
        chunks     : list of Chunk objects to index
        batch_size : upsert batch size
        """
        if not chunks:
            logger.warning("index() called with empty chunk list.")
            return

        logger.info(
            "Indexing %d chunks into collection '%s' …",
            len(chunks),
            self.collection_name,
        )

        # Encode all chunks
        embeddings = self._embed_model.encode_chunks(chunks)

        # Upsert in batches to avoid memory spikes
        for start in tqdm(range(0, len(chunks), batch_size), desc="Upserting to ChromaDB"):
            batch_chunks = chunks[start : start + batch_size]
            batch_vecs   = embeddings[start : start + batch_size]

            self._collection.upsert(
                ids=[c.chunk_id for c in batch_chunks],
                embeddings=batch_vecs.tolist(),
                documents=[c.text for c in batch_chunks],
                metadatas=[
                    {
                        "arxiv_id":    c.arxiv_id,
                        "title":       c.title,
                        "authors":     json.dumps(c.authors),
                        "year":        c.year,
                        "url":         c.url,
                        "strategy":    c.strategy,
                        "section":     c.section,
                        "chunk_index": c.chunk_index,
                    }
                    for c in batch_chunks
                ],
            )

        logger.info("Indexing complete. Collection size: %d", self._collection.count())

    def collection_size(self) -> int:
        """Return number of documents in the current collection."""
        return self._collection.count()

    # ── Retrieval ─────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k:    int | None = None,
        final_k:  int | None = None,
        threshold: float | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve and optionally re-rank the most relevant chunks for a query.

        Parameters
        ----------
        query     : natural-language question
        top_k     : candidates from ANN search (defaults to config)
        final_k   : chunks returned after re-ranking (defaults to config)
        threshold : minimum cosine similarity; results below are dropped

        Returns
        -------
        list of RetrievalResult, ordered by reranker score (or vector score
        if no re-ranker is enabled)
        """
        top_k    = top_k    or cfg["retrieval"]["top_k"]
        final_k  = final_k  or cfg["retrieval"]["final_k"]
        threshold = threshold if threshold is not None else cfg["retrieval"]["similarity_threshold"]

        if self._collection.count() == 0:
            logger.warning("Collection is empty — have you run index() first?")
            return []

        # 1. ANN search
        query_vec = self._embed_model.encode_query(query)
        raw = self._collection.query(
            query_embeddings=query_vec.tolist(),
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        # ChromaDB returns distances; convert to cosine similarity
        # distance metric is cosine → distance = 1 - similarity
        distances  = raw["distances"][0]
        documents  = raw["documents"][0]
        metadatas  = raw["metadatas"][0]
        chunk_ids  = raw["ids"][0]

        candidates: list[RetrievalResult] = []
        for cid, dist, doc, meta in zip(chunk_ids, distances, documents, metadatas):
            sim = max(0.0, 1.0 - dist)   # cosine similarity in [0, 1]
            chunk = Chunk(
                chunk_id=cid,
                text=doc,
                arxiv_id=meta["arxiv_id"],
                title=meta["title"],
                authors=json.loads(meta.get("authors", "[]")),
                year=int(meta.get("year", 0)),
                url=meta.get("url", ""),
                strategy=meta.get("strategy", ""),
                section=meta.get("section", ""),
                chunk_index=int(meta.get("chunk_index", 0)),
            )
            candidates.append(
                RetrievalResult(chunk=chunk, vector_score=sim, reranker_score=0.0, rank=-1)
            )

        # 2. Filter by threshold (applied before re-ranking to save compute)
        above = [r for r in candidates if r.vector_score >= threshold]
        if not above:
            logger.info("No chunks above similarity threshold %.2f for query: %s", threshold, query[:80])
            return []

        # 3. Cross-encoder re-ranking
        if self._reranker is not None:
            pairs = [(query, r.chunk.text) for r in above]
            scores = self._reranker.predict(pairs, show_progress_bar=False)
            for result, score in zip(above, scores):
                result.reranker_score = float(score)
            above.sort(key=lambda r: r.reranker_score, reverse=True)
        else:
            above.sort(key=lambda r: r.vector_score, reverse=True)

        # 4. Keep final_k and assign ranks
        final = above[:final_k]
        for rank, result in enumerate(final, start=1):
            result.rank = rank

        logger.debug(
            "Query: '%s' → %d candidates, %d above threshold, %d returned",
            query[:60],
            len(candidates),
            len(above),
            len(final),
        )
        return final

    def delete_collection(self) -> None:
        """Delete the current ChromaDB collection (useful for re-indexing)."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": cfg["chroma"]["distance_metric"]},
        )
        logger.info("Collection '%s' deleted and recreated.", self.collection_name)


# ── Convenience builder ───────────────────────────────────────────────────────

def build_retriever(
    chunks: list[Chunk],
    embedding_model_slug: str | None = None,
    chunking_strategy:    str | None = None,
    force_reindex:        bool = False,
) -> Retriever:
    """
    Build (or reload) a Retriever and index chunks if needed.

    Parameters
    ----------
    chunks               : list of Chunk objects
    embedding_model_slug : model slug (defaults to config)
    chunking_strategy    : strategy slug (defaults to config)
    force_reindex        : if True, wipe existing index and re-embed

    Returns
    -------
    Ready-to-query Retriever
    """
    retriever = Retriever(
        embedding_model_slug=embedding_model_slug,
        chunking_strategy=chunking_strategy,
    )
    if force_reindex or retriever.collection_size() < len(chunks):
        retriever.index(chunks)
    else:
        logger.info(
            "Reusing existing index with %d documents.", retriever.collection_size()
        )
    return retriever
