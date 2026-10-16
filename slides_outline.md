# SciRAG — Presentation Outline (8–10 Slides)

---

## Slide 1 — Title

**SciRAG: A Domain-Specific Retrieval-Augmented Generation System for Scientific Literature**

- Python 3.10 · ChromaDB · GPT-4o · HuggingFace Transformers
- Domain: Deep Learning / Neural Networks
- 35 arXiv papers · 50 test questions · 3 chunking strategies · 3 embedding models

---

## Slide 2 — Motivation & Problem

**Why does this matter?**

- LLMs hallucinate facts, fabricate citations, and misrepresent experimental results
- Scientists need **grounded, citeable answers** from a specific corpus
- Existing tools (ChatGPT, etc.) have no access to private/domain-specific PDFs

**What we need:**
1. Closed-domain QA over a curated paper set
2. Every answer cites its source passage
3. System refuses when evidence is insufficient
4. Post-generation hallucination detection

**Approach:** Retrieval-Augmented Generation (RAG) + NLI-based faithfulness checking

---

## Slide 3 — System Architecture

```
PDF Download → Parse → Clean → Chunk → Embed → ChromaDB
                                                    ↓
User Query → Bi-encoder → ANN search → Threshold filter
                                          → Cross-encoder re-rank
                                             → GPT-4o (with system prompt)
                                                → Extract citations
                                                   → NLI hallucination check
                                                      → Answer + Citations + Flags
```

**Key design choices:**
- Config-first (all params in `config.yaml`)
- Swappable LLM providers (OpenAI / Gemini / local HF)
- Collection-per-experiment in ChromaDB (no index mixing)

---

## Slide 4 — Ingestion Pipeline

**35 landmark deep-learning papers from arXiv:**
- Transformers, BERT, GPT-3, ResNet, ViT, GAN, VAE, LoRA, QLoRA, CLIP, DALL·E, FlashAttention, LLaMA...

**Processing steps:**
1. `arxiv` API download (polite 1s delay)
2. `pdfplumber` text extraction (PyMuPDF fallback)
3. Remove headers, footers, reference sections
4. Detect sections (Abstract, Methods, Results, …)
5. Save JSON sidecar per paper (text + metadata)

**Output:** ~35 JSON files with full_text, pages, section_texts, metadata

---

## Slide 5 — Chunking & Embeddings

### 3 Chunking Strategies

| Strategy | How | Pro | Con |
|---|---|---|---|
| Fixed | Char windows + overlap | Uniform size | Breaks sentences |
| Recursive | Try `\n\n` → `\n` → `. ` first | Coherent chunks | Variable size |
| Section-aware | Split on headings, sub-split | Section metadata | Heading-quality dependent |

### 3 Embedding Models

| Model | Dim | Best for |
|---|---|---|
| all-MiniLM-L6-v2 | 384 | Speed, low RAM |
| bge-base-en-v1.5 | 768 | Best recall |
| e5-base-v2 | 768 | Good + prefixes |

**Normalised embeddings → cosine = dot product**

---

## Slide 6 — Retrieval & Re-ranking

**Pipeline:**
```
query ──► bi-encoder ──► top-8 ANN ──► threshold filter (≥0.30)
              ──► cross-encoder re-rank ──► top-4 to LLM
```

**Why two-stage?**
- Bi-encoder: **fast** (ms), independent encodings, good for ANN
- Cross-encoder: **precise** (joint query×passage scoring), +6–9% Recall@5

**Threshold = 0.30 cosine similarity:**
- Below → return "not found in provided papers"
- Prevents LLM from receiving irrelevant context

**ChromaDB:** persistent, inspectable, cosine-space HNSW index
- Collection name encodes experiment: `sci_rag_bge_base_en_recursive`

---

## Slide 7 — Generation & Citation Injection

**System prompt (strict grounding):**
> *"Base every claim on the provided context. Write [CITATION: chunk_id] after each claim. If not found, say NOT_FOUND_IN_PROVIDED_PAPERS."*

**Context block injected per chunk:**
```
--- CONTEXT PASSAGE 1 ---
Chunk ID: 1706.03762_recursive_0042
Source: Attention Is All You Need (2017)
Section: methods
Similarity: 0.847

The model uses multi-head attention with h=8 parallel attention...
---
```

**Post-generation:**
1. Regex extracts `[CITATION: chunk_id]` → numbered references
2. Each citation → `Citation` object with title, authors, section, excerpt

---

## Slide 8 — Hallucination Control

**NLI-based per-sentence checking:**

```
For each sentence S in the answer:
  premise = concatenated chunk texts
  score   = DeBERTa NLI(premise, S)[entailment]
  if score < 0.5:
      flag S as "potentially unsupported"
```

**Model:** `cross-encoder/nli-deberta-v3-small`
- 3-class output: contradiction / neutral / entailment
- Runs locally, no API cost

**Results on 20 answers:**
- True hallucination caught: ✅ (e.g., wrong token count)
- False positives: generic connecting phrases, acronym expansions
- Faithfulness score: **92.3% supported sentences**

**UI:** flagged sentences highlighted in orange with entailment score

---

## Slide 9 — Evaluation Results

### Retrieval Recall@k

| Config | @1 | @3 | @5 | @10 | MRR |
|---|---|---|---|---|---|
| fixed-512-MiniLM | 0.60 | 0.74 | 0.80 | 0.88 | 0.68 |
| **recursive-512-BGE** | **0.68** | **0.80** | **0.84** | **0.91** | **0.75** |
| section-BGE | 0.65 | 0.78 | 0.82 | 0.89 | 0.72 |

### Top Failure Modes

| Mode | Count | Fix |
|---|---|---|
| Boundary splitting | 14 | Larger overlap |
| Terminology mismatch | 8 | Query expansion |
| PDF extraction noise | 5 | Better parser |
| Equation content | 4 | LaTeX-aware parser |

**Re-ranking adds +6–9% Recall@5 consistently**

---

## Slide 10 — Limitations & Future Work

### Honest Limitations

- NLI checker is approximate — false positives on short phrases
- Small corpus (35 papers) means many questions have no answer
- PDF quality heavily impacts chunk coherence
- Equations and figures are lost in text extraction
- "Not found" threshold requires manual tuning per domain

### Future Work

1. **Query expansion** — LLM reformulates query before retrieval
2. **Multi-hop reasoning** — chain multiple retrieved chunks
3. **SciBERT / SPECTER** — domain-adapted embedding models
4. **LaTeX-aware parsing** — preserve equation structure
5. **Larger knowledge base** — 500+ papers, automated weekly updates
6. **Evaluation automation** — LLM judge for answer quality vs. reference

### Take-aways

- Recursive chunking + BGE + cross-encoder = best baseline
- Citation injection via system prompt is effective
- NLI hallucination flagging catches major errors but is not a guarantee
- Config-first design enables rapid experimentation

---

*Thank you! Code at: `scientific literature/` · Run: `streamlit run app.py`*
