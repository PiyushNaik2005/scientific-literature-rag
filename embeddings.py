"""
embeddings.py — Unified embedding interface for three sentence-transformer models.

Supported models (configured in config.yaml)
---------------------------------------------
* all-MiniLM-L6-v2  (384-dim)  — fastest, lowest RAM
* bge-base-en-v1.5  (768-dim)  — strong general-purpose model
* e5-base-v2        (768-dim)  — requires "query: " / "passage: " prefixes

Design choices
--------------
* All models are wrapped in a single `EmbeddingModel` class so the rest of
  the pipeline is model-agnostic.
* Batched encoding (configurable batch_size) + tqdm progress bars for large
  corpora.
* Normalised embeddings (unit-length) are stored so that dot-product equals
  cosine similarity — ChromaDB's cosine metric then works correctly.
* The model slug (e.g. "bge-base-en-v1.5") is used to name ChromaDB
  collections, so switching models never overwrites an existing index.

Usage
-----
    from embeddings import get_embedding_model
    model = get_embedding_model("bge-base-en-v1.5")
    vecs  = model.encode_passages(["hello world", "foo bar"])
    query = model.encode_query("what is attention?")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from chunking import Chunk
from utils import fix_seed, get_config, get_logger

logger = get_logger(__name__)
cfg = get_config()

# ── Model wrapper ─────────────────────────────────────────────────────────────

class EmbeddingModel:
    """
    Wraps a SentenceTransformer with model-specific prefix handling
    and batched normalised encoding.

    Parameters
    ----------
    model_slug : key in config.yaml embeddings.models (e.g. "bge-base-en-v1.5")
    """

    def __init__(self, model_slug: str) -> None:
        model_cfgs = cfg["embeddings"]["models"]
        if model_slug not in model_cfgs:
            raise ValueError(
                f"Unknown model '{model_slug}'. Available: {list(model_cfgs)}"
            )

        self.slug       = model_slug
        self.model_cfg  = model_cfgs[model_slug]
        self.hf_name    = self.model_cfg["hf_name"]
        self.dimension  = self.model_cfg["dimension"]
        self.batch_size = self.model_cfg["batch_size"]

        # e5 models require prefixes; others don't
        self.query_prefix   = self.model_cfg.get("query_prefix",   "")
        self.passage_prefix = self.model_cfg.get("passage_prefix", "")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading embedding model '%s' on %s …", self.hf_name, device)
        self._model = SentenceTransformer(self.hf_name, device=device)
        logger.info("Embedding model ready (dim=%d).", self.dimension)

    # ── Core encoding ──────────────────────────────────────────────────────

    def _encode(
        self,
        texts: list[str],
        prefix: str = "",
        desc: str = "Encoding",
    ) -> np.ndarray:
        """
        Encode a list of texts in batches and return L2-normalised vectors.

        Parameters
        ----------
        texts  : input strings
        prefix : prepended to each text (for e5 models)
        desc   : tqdm description

        Returns
        -------
        np.ndarray of shape (len(texts), dimension), float32, unit-norm rows
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        prefixed = [prefix + t for t in texts] if prefix else texts

        all_vecs: list[np.ndarray] = []
        for start in tqdm(
            range(0, len(prefixed), self.batch_size),
            desc=desc,
            unit="batch",
            leave=False,
        ):
            batch = prefixed[start : start + self.batch_size]
            vecs = self._model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,   # unit-norm → cosine = dot product
                show_progress_bar=False,
            )
            all_vecs.append(vecs.astype(np.float32))

        return np.vstack(all_vecs)

    # ── Public API ─────────────────────────────────────────────────────────

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of document passages (for indexing).

        Returns
        -------
        np.ndarray shape (N, dim), float32, unit-norm
        """
        return self._encode(texts, prefix=self.passage_prefix, desc=f"Encoding passages [{self.slug}]")

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string.

        Returns
        -------
        np.ndarray shape (1, dim), float32, unit-norm
        """
        return self._encode([query], prefix=self.query_prefix, desc="Encoding query")

    def encode_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        """
        Encode the text of a list of Chunk objects.

        Returns
        -------
        np.ndarray shape (N, dim), float32, unit-norm
        """
        texts = [c.text for c in chunks]
        return self.encode_passages(texts)


# ── Factory / cache ───────────────────────────────────────────────────────────

_MODEL_CACHE: dict[str, EmbeddingModel] = {}


def get_embedding_model(model_slug: str | None = None) -> EmbeddingModel:
    """
    Return (and cache) an EmbeddingModel instance.

    Parameters
    ----------
    model_slug : model key from config (defaults to embeddings.default_model)

    Returns
    -------
    EmbeddingModel
    """
    if model_slug is None:
        model_slug = cfg["embeddings"]["default_model"]
    if model_slug not in _MODEL_CACHE:
        fix_seed()
        _MODEL_CACHE[model_slug] = EmbeddingModel(model_slug)
    return _MODEL_CACHE[model_slug]


# ── Comparison utility ────────────────────────────────────────────────────────

def compare_models(
    texts: list[str],
    query: str,
    model_slugs: list[str] | None = None,
) -> dict[str, dict]:
    """
    Encode texts with multiple models and return similarity rankings.

    Useful for the evaluation grid (chunk_size × embedding_model).

    Parameters
    ----------
    texts       : list of passage texts
    query       : query string
    model_slugs : models to compare (defaults to all configured models)

    Returns
    -------
    dict mapping model_slug → {"similarities": [float], "top_indices": [int]}
    """
    if model_slugs is None:
        model_slugs = list(cfg["embeddings"]["models"].keys())

    results = {}
    for slug in model_slugs:
        model = get_embedding_model(slug)
        passage_vecs = model.encode_passages(texts)
        query_vec    = model.encode_query(query)

        # cosine similarity (already normalised, so dot product)
        sims = (passage_vecs @ query_vec.T).squeeze().tolist()
        if isinstance(sims, float):
            sims = [sims]

        top_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        results[slug] = {"similarities": sims, "top_indices": top_indices}

    return results
