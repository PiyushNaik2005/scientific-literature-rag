"""
evaluate.py — Evaluation suite for the RAG pipeline.

Metrics
-------
* Retrieval hit rate / Recall@k  — fraction of queries where the correct
  paper appears in the top-k retrieved chunks
* MRR (Mean Reciprocal Rank)     — measures how high the first relevant
  result appears in the ranking
* Answer faithfulness            — fraction of answer sentences supported
  by the retrieved context (NLI-based)
* Chunk-size × model grid        — compares all combinations and produces
  a results table + chart

Failure analysis
----------------
Every retrieval failure (correct paper not in top-k) and every flagged
hallucination is logged with the query, expected paper, retrieved papers,
and a diagnosis note.

Usage
-----
    python evaluate.py                  # full evaluation with default config
    python evaluate.py --k 5            # evaluate recall@5 only
    python evaluate.py --grid           # run full comparison grid
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from chunking import chunk_all_papers, get_chunker
from embeddings import get_embedding_model
from generator import Generator
from ingest import load_processed
from retriever import Retriever, build_retriever
from utils import fix_seed, get_config, get_logger

logger = get_logger(__name__)
cfg = get_config()

# ── Test Question Bank ────────────────────────────────────────────────────────
# 50 questions with known source paper (arxiv_id) for reproducible evaluation.
# Format: {question, expected_arxiv_id, expected_keywords_in_answer}

DEFAULT_TEST_QUESTIONS: list[dict] = [
    # Transformer / Attention
    {"question": "What is the core innovation of the Transformer architecture?", "expected_arxiv_id": "1706.03762", "keywords": ["attention", "self-attention", "encoder", "decoder"]},
    {"question": "How does multi-head attention differ from single-head attention?", "expected_arxiv_id": "1706.03762", "keywords": ["multi-head", "parallel", "subspaces"]},
    {"question": "What positional encoding scheme is used in the original Transformer?", "expected_arxiv_id": "1706.03762", "keywords": ["sinusoidal", "positional", "encoding"]},
    {"question": "What is the feed-forward sublayer in the Transformer block?", "expected_arxiv_id": "1706.03762", "keywords": ["feed-forward", "position-wise", "relu"]},
    # BERT
    {"question": "What pretraining objectives does BERT use?", "expected_arxiv_id": "1810.04805", "keywords": ["masked language model", "next sentence prediction"]},
    {"question": "How does BERT handle bidirectionality compared to GPT?", "expected_arxiv_id": "1810.04805", "keywords": ["bidirectional", "left-to-right", "context"]},
    {"question": "What is the WordPiece tokenization used in BERT?", "expected_arxiv_id": "1810.04805", "keywords": ["wordpiece", "subword", "vocabulary"]},
    # GPT-3
    {"question": "What is few-shot learning as described in the GPT-3 paper?", "expected_arxiv_id": "2005.14165", "keywords": ["few-shot", "in-context", "examples"]},
    {"question": "How large is the GPT-3 model in terms of parameters?", "expected_arxiv_id": "2005.14165", "keywords": ["175 billion", "parameters"]},
    # Vision Transformer
    {"question": "How does the Vision Transformer (ViT) process image patches?", "expected_arxiv_id": "2010.11929", "keywords": ["patch", "linear projection", "sequence"]},
    {"question": "What dataset size is required for ViT to match CNNs?", "expected_arxiv_id": "2010.11929", "keywords": ["large dataset", "JFT", "ImageNet-21k"]},
    # ResNet
    {"question": "What is the residual connection in ResNet?", "expected_arxiv_id": "1512.03385", "keywords": ["skip connection", "identity", "shortcut"]},
    {"question": "How deep is the deepest ResNet variant in the original paper?", "expected_arxiv_id": "1512.03385", "keywords": ["152", "layers", "deep"]},
    # Batch Normalization
    {"question": "What problem does Batch Normalization address?", "expected_arxiv_id": "1502.03167", "keywords": ["internal covariate shift", "normalization"]},
    {"question": "How does batch normalization affect training speed?", "expected_arxiv_id": "1502.03167", "keywords": ["faster", "learning rate", "training"]},
    # Adam
    {"question": "How does the Adam optimizer combine momentum and RMSProp?", "expected_arxiv_id": "1412.6980", "keywords": ["first moment", "second moment", "adaptive"]},
    {"question": "What are the default hyperparameter values for Adam?", "expected_arxiv_id": "1412.6980", "keywords": ["beta1", "beta2", "epsilon"]},
    # VAE
    {"question": "What is the reparameterization trick in VAEs?", "expected_arxiv_id": "1312.6114", "keywords": ["reparameterization", "sampling", "gradient"]},
    {"question": "What is the ELBO objective in variational autoencoders?", "expected_arxiv_id": "1312.6114", "keywords": ["evidence lower bound", "KL divergence", "reconstruction"]},
    # GAN
    {"question": "What is the minimax game in the original GAN paper?", "expected_arxiv_id": "1406.2661", "keywords": ["generator", "discriminator", "minimax", "adversarial"]},
    {"question": "What is the mode collapse problem in GANs?", "expected_arxiv_id": "1406.2661", "keywords": ["mode", "collapse", "diversity"]},
    # Dropout
    {"question": "How does dropout prevent overfitting in neural networks?", "expected_arxiv_id": "1207.0580", "keywords": ["dropout", "regularization", "co-adaptation"]},
    # T5
    {"question": "How does T5 frame all NLP tasks as text-to-text?", "expected_arxiv_id": "1910.10683", "keywords": ["text-to-text", "unified", "prefix"]},
    # LoRA
    {"question": "What is the key idea behind LoRA for efficient fine-tuning?", "expected_arxiv_id": "2106.09685", "keywords": ["low-rank", "decomposition", "adapter", "frozen"]},
    {"question": "How does LoRA reduce the number of trainable parameters?", "expected_arxiv_id": "2106.09685", "keywords": ["rank", "matrices", "parameters"]},
    # CLIP
    {"question": "How does CLIP learn image-text representations?", "expected_arxiv_id": "2103.00020", "keywords": ["contrastive", "image-text", "pairs"]},
    {"question": "What training dataset does CLIP use?", "expected_arxiv_id": "2103.00020", "keywords": ["400 million", "web", "image-text"]},
    # Latent Diffusion
    {"question": "What is the key contribution of Latent Diffusion Models?", "expected_arxiv_id": "2112.10752", "keywords": ["latent space", "compression", "perceptual"]},
    # MAE
    {"question": "What masking ratio does MAE use during pretraining?", "expected_arxiv_id": "2111.11418", "keywords": ["75%", "masking ratio", "high"]},
    # InstructGPT / RLHF
    {"question": "What is RLHF and how is it used to align language models?", "expected_arxiv_id": "2203.02155", "keywords": ["reinforcement learning", "human feedback", "reward model"]},
    # PPO
    {"question": "What is the clipped surrogate objective in PPO?", "expected_arxiv_id": "1707.06347", "keywords": ["clipped", "surrogate", "policy"]},
    # GCN
    {"question": "How does Graph Convolutional Network aggregate neighbor features?", "expected_arxiv_id": "1609.02907", "keywords": ["aggregation", "neighbors", "spectral"]},
    # Scaling Laws
    {"question": "What does the scaling law paper find about compute-optimal training?", "expected_arxiv_id": "2001.08361", "keywords": ["scaling", "power law", "compute"]},
    # Chinchilla
    {"question": "What is the Chinchilla finding about model size vs tokens?", "expected_arxiv_id": "2203.15556", "keywords": ["compute optimal", "tokens", "model size"]},
    # FlashAttention
    {"question": "How does FlashAttention reduce memory usage?", "expected_arxiv_id": "2205.01068", "keywords": ["IO-aware", "tiling", "memory"]},
    # QLoRA
    {"question": "What quantization technique does QLoRA use?", "expected_arxiv_id": "2305.10403", "keywords": ["4-bit", "NF4", "quantization"]},
    # ELMo
    {"question": "How do ELMo embeddings differ from word2vec embeddings?", "expected_arxiv_id": "1803.05457", "keywords": ["contextualized", "bidirectional", "character"]},
    # RoBERTa
    {"question": "What training differences does RoBERTa make compared to BERT?", "expected_arxiv_id": "1909.11942", "keywords": ["longer training", "more data", "NSP removed"]},
    # DistilBERT
    {"question": "How does DistilBERT reduce BERT's size while keeping performance?", "expected_arxiv_id": "1910.01108", "keywords": ["knowledge distillation", "smaller", "teacher"]},
    # LLaMA
    {"question": "What is the architecture of the original LLaMA model?", "expected_arxiv_id": "2302.13971", "keywords": ["transformer", "rotary", "SwiGLU"]},
    # LLaMA 2
    {"question": "What safety training is applied to LLaMA 2?", "expected_arxiv_id": "2307.09288", "keywords": ["RLHF", "safety", "fine-tuning", "chat"]},
    # DeiT
    {"question": "How does DeiT train vision transformers without external data?", "expected_arxiv_id": "2010.09700", "keywords": ["distillation", "token", "ImageNet"]},
    # GAT
    {"question": "How does Graph Attention Network assign weights to neighbors?", "expected_arxiv_id": "1710.10903", "keywords": ["attention", "neighbors", "coefficients"]},
    # VGG
    {"question": "What is the architectural principle of VGGNet?", "expected_arxiv_id": "1409.1556", "keywords": ["3x3", "convolution", "depth", "small filters"]},
    # RoFormer
    {"question": "What is Rotary Position Embedding (RoPE)?", "expected_arxiv_id": "2104.09864", "keywords": ["rotary", "position", "rotation", "relative"]},
    # DALL-E 1
    {"question": "How does the original DALL-E generate images from text?", "expected_arxiv_id": "2010.14701", "keywords": ["discrete VAE", "tokens", "transformer", "text"]},
    # DALL-E 2
    {"question": "What role does CLIP play in DALL-E 2?", "expected_arxiv_id": "2204.02311", "keywords": ["CLIP", "image embedding", "prior"]},
    # LM-BFF / PEFT overview
    {"question": "What is parameter-efficient fine-tuning (PEFT)?", "expected_arxiv_id": "2209.01188", "keywords": ["adapter", "prefix", "efficient", "fine-tuning"]},
    # Curriculum Learning
    {"question": "What is the main idea behind curriculum learning in training?", "expected_arxiv_id": "1608.05343", "keywords": ["curriculum", "easy", "hard", "order"]},
]


# ── Metrics ───────────────────────────────────────────────────────────────────

def recall_at_k(results: list[dict], k: int) -> float:
    """
    Fraction of queries where the expected paper appears in the top-k results.

    Parameters
    ----------
    results : list of {"expected_arxiv_id": str, "retrieved_ids": [str]}
    k       : cutoff

    Returns
    -------
    float in [0, 1]
    """
    hits = sum(
        1
        for r in results
        if r["expected_arxiv_id"] in r["retrieved_ids"][:k]
    )
    return hits / len(results) if results else 0.0


def mrr(results: list[dict]) -> float:
    """
    Mean Reciprocal Rank across all queries.

    Parameters
    ----------
    results : list of {"expected_arxiv_id": str, "retrieved_ids": [str]}

    Returns
    -------
    float in [0, 1]
    """
    reciprocal_ranks = []
    for r in results:
        try:
            rank = r["retrieved_ids"].index(r["expected_arxiv_id"]) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0.0)
    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


def faithfulness_score(flagged_counts: list[int], total_sentences: list[int]) -> float:
    """
    Fraction of answer sentences that are well-supported.

    Parameters
    ----------
    flagged_counts   : number of flagged (unsupported) sentences per answer
    total_sentences  : total sentences per answer

    Returns
    -------
    float in [0, 1]
    """
    if not total_sentences:
        return 0.0
    supported = sum(t - f for t, f in zip(total_sentences, flagged_counts))
    total     = sum(total_sentences)
    return supported / total if total > 0 else 0.0


def normalize_arxiv_id(aid: str) -> str:
    """Normalize arXiv ID by removing version suffix, e.g. 1706.03762v7 -> 1706.03762."""
    import re
    return re.sub(r"v\d+$", "", aid.strip())


# ── Evaluator class ───────────────────────────────────────────────────────────

class Evaluator:
    """
    Runs the full evaluation suite and saves results.

    Parameters
    ----------
    retriever  : configured and indexed Retriever
    generator  : configured Generator
    eval_dir   : directory to write results
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        eval_dir: Path,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.eval_dir  = eval_dir
        self.eval_dir.mkdir(parents=True, exist_ok=True)

    def load_questions(self, qa_file: Path | None = None) -> list[dict]:
        """Load test questions from JSON file or use defaults."""
        if qa_file and qa_file.exists():
            with open(qa_file, "r", encoding="utf-8") as f:
                return json.load(f)
        logger.info("Using built-in test question bank (%d questions).", len(DEFAULT_TEST_QUESTIONS))
        return DEFAULT_TEST_QUESTIONS

    def run_retrieval_eval(
        self,
        questions: list[dict],
        k_values: list[int] | None = None,
    ) -> dict:
        """
        Evaluate retrieval quality across multiple k values.

        Returns
        -------
        dict with recall@k for each k, MRR, and per-query details
        """
        k_values = k_values or cfg["evaluation"]["recall_k_values"]
        max_k = max(k_values)

        per_query_results = []
        failure_log       = []

        for q in tqdm(questions, desc="Retrieval eval"):
            retrieved = self.retriever.retrieve(q["question"], top_k=max_k, final_k=max_k)
            retrieved_arxiv_ids = [r.chunk.arxiv_id for r in retrieved]
            # Deduplicate while preserving order
            seen = set()
            unique_ids = []
            for aid in retrieved_arxiv_ids:
                if aid not in seen:
                    seen.add(aid)
                    unique_ids.append(aid)

            clean_expected = normalize_arxiv_id(q["expected_arxiv_id"])
            clean_retrieved = [normalize_arxiv_id(aid) for aid in unique_ids]
            hit = clean_expected in clean_retrieved[:max_k]

            record = {
                "question":           q["question"],
                "expected_arxiv_id":  clean_expected,
                "retrieved_ids":      clean_retrieved,
                "hit":                hit,
            }
            per_query_results.append(record)

            if not hit:
                failure_log.append(
                    {
                        "question":          q["question"],
                        "expected":          clean_expected,
                        "top_retrieved":     clean_retrieved[:5],
                        "top_scores":        [r.vector_score for r in retrieved[:5]],
                        "diagnosis":         self._diagnose_retrieval_failure(q, retrieved),
                    }
                )

        metrics = {
            f"recall@{k}": recall_at_k(per_query_results, k) for k in k_values
        }
        metrics["mrr"]          = mrr(per_query_results)
        metrics["n_questions"]  = len(questions)
        metrics["n_failures"]   = len(failure_log)

        # Save outputs
        self._save_json(per_query_results, "retrieval_per_query.json")
        self._save_json(failure_log,       "retrieval_failures.json")
        self._save_json(metrics,           "retrieval_metrics.json")

        logger.info("Retrieval metrics: %s", metrics)
        return metrics

    def _diagnose_retrieval_failure(self, question: dict, retrieved) -> str:
        """Heuristic failure diagnosis for the failure log."""
        if not retrieved:
            return "No chunks retrieved (all below similarity threshold)"
        top_paper = retrieved[0].chunk.arxiv_id if retrieved else "none"
        return (
            f"Top result was '{top_paper}' with score {retrieved[0].vector_score:.3f}. "
            "Possible causes: (1) expected paper not in knowledge base, "
            "(2) query phrasing too different from chunk text, "
            "(3) relevant chunk was split across boundary."
        )

    def run_faithfulness_eval(
        self,
        questions: list[dict],
        sample_size: int | None = None,
    ) -> dict:
        """
        Evaluate answer faithfulness (fraction of supported sentences).

        Parameters
        ----------
        questions   : test questions (uses a random sample to save API cost)
        sample_size : number of questions to evaluate

        Returns
        -------
        dict with faithfulness score and per-answer details
        """
        sample_size = sample_size or cfg["evaluation"]["faithfulness_sample_size"]
        rng = np.random.default_rng(cfg["seed"])
        sampled = rng.choice(questions, size=min(sample_size, len(questions)), replace=False).tolist()

        per_answer = []
        flagged_counts  = []
        total_sentences = []

        for q in tqdm(sampled, desc="Faithfulness eval"):
            retrieved = self.retriever.retrieve(q["question"])
            result    = self.generator.generate(q["question"], retrieved)

            n_sentences = len([s for s in result.answer.split(". ") if len(s.split()) >= 5])
            n_flagged   = len(result.flagged_sentences)

            flagged_counts.append(n_flagged)
            total_sentences.append(n_sentences)

            per_answer.append(
                {
                    "question":           q["question"],
                    "answer":             result.answer,
                    "n_sentences":        n_sentences,
                    "n_flagged":          n_flagged,
                    "flagged_sentences":  result.flagged_sentences,
                    "not_found":          result.not_found,
                }
            )

        faith = faithfulness_score(flagged_counts, total_sentences)
        metrics = {
            "faithfulness": faith,
            "n_evaluated":  len(sampled),
            "avg_flagged":  float(np.mean(flagged_counts)) if flagged_counts else 0.0,
        }

        self._save_json(per_answer, "faithfulness_per_answer.json")
        self._save_json(metrics,    "faithfulness_metrics.json")

        logger.info("Faithfulness: %.3f", faith)
        return metrics

    # ── Results table & chart ──────────────────────────────────────────────

    def run_comparison_grid(
        self,
        questions: list[dict],
        embedding_slugs: list[str] | None = None,
        strategies:     list[str] | None = None,
        chunk_sizes:    list[int]  | None = None,
        k: int = 5,
    ) -> pd.DataFrame:
        """
        Grid search over chunk sizes and embedding models.

        Returns
        -------
        DataFrame with rows = (strategy, chunk_size, model) and
        columns = recall@k, MRR
        """
        if embedding_slugs is None:
            embedding_slugs = list(cfg["embeddings"]["models"].keys())
        if strategies is None:
            strategies = ["fixed", "recursive", "section"]
        if chunk_sizes is None:
            chunk_sizes = [256, 512, 1024]

        root         = Path(__file__).parent
        processed_dir = root / cfg["paths"]["processed_dir"]
        papers       = load_processed(processed_dir)

        rows = []
        for strategy in strategies:
            for size in chunk_sizes:
                chunks = chunk_all_papers(papers, strategy=strategy, chunk_size=size, overlap=size // 8)
                for slug in embedding_slugs:
                    logger.info("Grid: strategy=%s size=%d model=%s", strategy, size, slug)
                    try:
                        retriever = build_retriever(chunks, embedding_model_slug=slug, chunking_strategy=strategy, force_reindex=True)
                        per_q = []
                        for q in questions[:20]:  # subset for speed
                            retrieved = retriever.retrieve(q["question"], top_k=k, final_k=k)
                            ids = list({r.chunk.arxiv_id for r in retrieved})
                            per_q.append({"expected_arxiv_id": q["expected_arxiv_id"], "retrieved_ids": ids})
                        rows.append(
                            {
                                "strategy":   strategy,
                                "chunk_size": size,
                                "model":      slug,
                                f"recall@{k}": recall_at_k(per_q, k),
                                "mrr":         mrr(per_q),
                            }
                        )
                    except Exception as exc:
                        logger.warning("Grid cell failed: %s", exc)

        df = pd.DataFrame(rows)
        df.to_csv(self.eval_dir / "grid_results.csv", index=False)
        self._plot_grid(df, k=k)
        logger.info("Grid results saved.")
        return df

    def _plot_grid(self, df: pd.DataFrame, k: int = 5) -> None:
        """Save a heatmap of recall@k across models and chunk sizes."""
        try:
            fig, axes = plt.subplots(1, len(df["strategy"].unique()), figsize=(14, 5), squeeze=False)
            for ax, strategy in zip(axes[0], df["strategy"].unique()):
                subset = df[df["strategy"] == strategy]
                pivot  = subset.pivot(index="chunk_size", columns="model", values=f"recall@{k}")
                sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax, vmin=0, vmax=1)
                ax.set_title(f"Recall@{k} — {strategy} chunking")
                ax.set_xlabel("Embedding Model")
                ax.set_ylabel("Chunk Size")
            plt.tight_layout()
            path = self.eval_dir / f"grid_recall_at_{k}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info("Grid heatmap saved → %s", path)
        except Exception as exc:
            logger.warning("Could not generate grid plot: %s", exc)

    # ── Utility ────────────────────────────────────────────────────────────

    def _save_json(self, data, filename: str) -> None:
        path = self.eval_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        logger.debug("Saved → %s", path)


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Scientific RAG pipeline")
    parser.add_argument("--k",      type=int, default=5,   help="Recall@k cutoff")
    parser.add_argument("--grid",   action="store_true",   help="Run full comparison grid")
    parser.add_argument("--no-gen", action="store_true",   help="Skip faithfulness eval (no LLM calls)")
    args = parser.parse_args()

    fix_seed()

    root          = Path(__file__).parent
    processed_dir = root / cfg["paths"]["processed_dir"]
    eval_dir      = root / cfg["paths"]["eval_dir"]

    papers = load_processed(processed_dir)
    if not papers:
        logger.error("No processed papers found. Run ingest.py first.")
        return

    # Build default retriever
    from chunking import chunk_all_papers
    strategy = cfg["chunking"]["default_strategy"]
    chunks   = chunk_all_papers(papers, strategy=strategy)
    retriever = build_retriever(chunks)
    generator = Generator() if not args.no_gen else None

    evaluator = Evaluator(retriever, generator, eval_dir)
    questions = evaluator.load_questions(
        Path(cfg["evaluation"]["qa_file"]) if Path(cfg["evaluation"]["qa_file"]).exists() else None
    )

    logger.info("=== Retrieval Evaluation ===")
    ret_metrics = evaluator.run_retrieval_eval(questions, k_values=[1, 3, 5, 10])

    if not args.no_gen and generator:
        logger.info("=== Faithfulness Evaluation ===")
        faith_metrics = evaluator.run_faithfulness_eval(questions)

    if args.grid:
        logger.info("=== Comparison Grid ===")
        df = evaluator.run_comparison_grid(questions, k=args.k)
        print("\n=== Grid Results ===")
        print(df.to_string(index=False))

    # Print summary
    print("\n=== Evaluation Summary ===")
    for key, val in ret_metrics.items():
        if isinstance(val, float):
            print(f"  {key:20s}: {val:.4f}")
        else:
            print(f"  {key:20s}: {val}")


if __name__ == "__main__":
    main()
