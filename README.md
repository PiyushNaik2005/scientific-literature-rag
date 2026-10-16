# 🔬 SciRAG — Scientific Literature RAG System

A production-quality **Retrieval-Augmented Generation** system for answering questions about deep learning research papers. Every answer is grounded in retrieved source text with inline citations, and a post-generation hallucination detector flags unsupported claims.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           INGESTION PIPELINE                            │
│  arXiv API ──► PDF Download ──► pdfplumber/PyMuPDF ──► Text + Metadata │
│                                      │                                   │
│                              Remove headers/                             │
│                              footers/references                         │
│                                      │                                   │
│                           Save JSON sidecar                             │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────────┐
│                         CHUNKING STRATEGIES                              │
│  FixedChunker ──────────────────────────────── char windows + overlap   │
│  RecursiveChunker ──────────────────────────── \n\n → \n → ". " → …   │
│  SectionChunker ────────────────────────────── header regex → sub-split │
│                                                                          │
│  Each chunk carries: title · authors · year · section · arxiv_id        │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────────┐
│                        EMBEDDING + INDEXING                              │
│  all-MiniLM-L6-v2  (384d) ─┐                                           │
│  bge-base-en-v1.5  (768d) ──┼──► Normalised vectors ──► ChromaDB       │
│  e5-base-v2        (768d) ─┘         (cosine space)                    │
│                                                                          │
│  Collection name: sci_rag_{model}_{strategy}  (isolated per experiment) │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────────┐
│                           RETRIEVAL                                      │
│                                                                          │
│  User query ──► bi-encoder ──► ANN search (top_k=8)                    │
│                      │                                                   │
│                threshold filter (cosine ≥ 0.30)                        │
│                      │                                                   │
│             cross-encoder re-rank ──► final_k=4 chunks                 │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────────┐
│                      GENERATION + CITATION                               │
│                                                                          │
│  Chunks ──► prompt with [CITATION: chunk_id] instructions               │
│         ──► GPT-4o / Gemini / Local HF model                           │
│         ──► extract [CITATION:...] markers ──► structured Citations     │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────────┐
│                   HALLUCINATION DETECTION                                │
│                                                                          │
│  Split answer into sentences ──► NLI DeBERTa cross-encoder             │
│  premise = concatenated chunk texts                                      │
│  entailment score < 0.5 ──► sentence flagged as unsupported            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
scientific-literature-rag/
├── app.py            # Streamlit UI (4 pages)
├── ingest.py         # PDF download, parsing, and processing
├── chunking.py       # 3 chunking strategies
├── embeddings.py     # 3 embedding model wrappers
├── retriever.py      # ChromaDB vector store + cross-encoder re-ranking
├── generator.py      # LLM generation with citation injection
├── evaluate.py       # Recall@k, MRR, faithfulness, comparison grid
├── utils.py          # Config loading, logging, seed fixing
├── config.yaml       # All hyperparameters (no Python edits needed)
├── requirements.txt
├── .env.example      # API key template
├── .gitignore
└── data/
    ├── papers/       # raw PDF downloads
    ├── processed/    # JSON sidecars (text + metadata)
    ├── chroma_db/    # ChromaDB persistent store
    └── evaluation/   # metrics, logs, charts
```

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- 8–16 GB RAM recommended (more for local HF models)
- GPU optional but beneficial for embedding large corpora

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your API key

```bash
cp .env.example .env
# Edit .env and fill in OPENAI_API_KEY
```

### 4. Download and process papers (~10–20 minutes)

```bash
python ingest.py --n 35
```

This downloads 35 landmark deep-learning papers from arXiv, extracts text, removes headers/footers/references, detects sections, and saves JSON sidecars to `data/processed/`.

### 5. Launch the UI

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501). On first load, the app embeds all chunks and builds the ChromaDB index (~5 minutes for 35 papers with MiniLM).

---

## Configuration

All parameters live in [`config.yaml`](config.yaml). Key settings:

| Section | Key | Default | Effect |
|---|---|---|---|
| `chunking` | `default_strategy` | `recursive` | Which chunker to use |
| `chunking.recursive` | `chunk_size` | `512` | Characters per chunk |
| `embeddings` | `default_model` | `all-MiniLM-L6-v2` | Embedding model |
| `retrieval` | `top_k` | `8` | ANN candidates |
| `retrieval` | `final_k` | `4` | Chunks sent to LLM |
| `retrieval` | `similarity_threshold` | `0.30` | Min cosine similarity |
| `retrieval` | `use_reranker` | `true` | Cross-encoder re-ranking |
| `generation` | `llm_provider` | `openai` | `openai`/`gemini`/`huggingface` |
| `hallucination` | `entailment_threshold` | `0.5` | NLI flag threshold |

---

## Running Evaluations

```bash
# Basic retrieval evaluation (recall@k + MRR)
python evaluate.py

# Skip faithfulness (no LLM calls, faster)
python evaluate.py --no-gen

# Full comparison grid (all chunk sizes × all models)
python evaluate.py --grid --no-gen
```

Results are saved to `data/evaluation/`:
- `retrieval_metrics.json` — recall@k and MRR summary
- `retrieval_per_query.json` — per-question results
- `retrieval_failures.json` — failure log with diagnoses
- `faithfulness_metrics.json` — NLI-based faithfulness score
- `grid_results.csv` — comparison table
- `grid_recall_at_5.png` — heatmap chart

---

## Results (Example — will vary with your papers)

### Retrieval (recursive chunking, bge-base-en-v1.5, top-k=5)

| Metric | Value |
|---|---|
| Recall@1 | 0.62 |
| Recall@3 | 0.78 |
| Recall@5 | 0.84 |
| Recall@10 | 0.91 |
| MRR | 0.71 |

### Faithfulness

| Metric | Value |
|---|---|
| Supported sentences | 92.3% |
| Avg flagged per answer | 0.4 |

### Chunk Size × Embedding Model Grid (Recall@5)

| Strategy | Chunk Size | MiniLM | BGE | E5 |
|---|---|---|---|---|
| fixed | 256 | 0.65 | 0.71 | 0.69 |
| fixed | 512 | 0.72 | 0.79 | 0.76 |
| fixed | 1024 | 0.68 | 0.74 | 0.72 |
| recursive | 256 | 0.70 | 0.76 | 0.73 |
| **recursive** | **512** | **0.78** | **0.84** | **0.81** |
| recursive | 1024 | 0.71 | 0.77 | 0.75 |
| section | — | 0.74 | 0.81 | 0.78 |

*Best configuration: recursive + 512 chars + bge-base-en-v1.5*

---

## Limitations

1. **PDF quality dependency** — scanned PDFs, two-column layouts, and papers with heavy equations produce noisier text extraction, which harms both chunking coherence and retrieval accuracy.

2. **Chunking boundary problem** — answers that span a section break are split across chunks; the retriever may find half the answer but not the other half. Section-aware chunking mitigates this but doesn't eliminate it.

3. **Hallucination detector is approximate** — the DeBERTa NLI model occasionally misclassifies correctly-cited sentences as unsupported (false positives) and misses subtle hallucinations (false negatives). It is a heuristic, not a guarantee.

4. **Cross-lingual and domain mismatch** — embedding models trained on general text perform worse on heavily mathematical or domain-specific passages. Specialised models (e.g., `allenai/scibert_scivocab_uncased`) may improve results but were not benchmarked here.

5. **Cost of faithfulness evaluation** — the full faithfulness eval makes one LLM call per question. For 50 questions at GPT-4o pricing, this costs ~$0.50–1.00.

6. **Knowledge cutoff** — the system knows only what is in the indexed papers. Questions requiring knowledge synthesis across papers may receive incomplete answers.

7. **Section detection heuristics** — the section detector uses simple regex patterns. Unusual heading formats (e.g., numbered sections with Roman numerals, non-English papers) may not be detected, falling back to `preamble`.

---

## Design Choices

| Choice | Rationale |
|---|---|
| ChromaDB over FAISS | Persistent, queryable, easy to inspect; FAISS is faster but in-memory only |
| Cross-encoder re-ranker | Consistently improves precision at minimal cost vs. bi-encoder alone |
| NLI-based hallucination detection | Model-agnostic; works without LLM API calls |
| Three chunking strategies | Different paper structures benefit from different strategies; grid search finds the best |
| Normalised embeddings | Dot product equals cosine similarity, which is what ChromaDB's `cosine` metric uses |
| Config-first design | All hyperparameters in `config.yaml`; no Python edits needed for experiments |

---

## References

1. Vaswani et al. (2017). *Attention Is All You Need.* arXiv:1706.03762
2. Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers.* arXiv:1810.04805
3. Brown et al. (2020). *Language Models are Few-Shot Learners.* arXiv:2005.14165
4. Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* arXiv:2005.11401
5. Reimers & Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* arXiv:1908.10084
6. Nogueira & Cho (2019). *Passage Re-ranking with BERT.* arXiv:1901.04085
7. He et al. (2021). *DeBERTa: Decoding-enhanced BERT with Disentangled Attention.* arXiv:2006.03654
8. Johnson et al. (2019). *Billion-scale similarity search with GPUs.* arXiv:1702.08734
