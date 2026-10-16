"""
ingest.py — Download arXiv papers and extract structured text + metadata.

Pipeline:
  1. Query arXiv API for deep-learning papers (35 papers by default).
  2. Download PDFs to data/papers/.
  3. Parse each PDF with pdfplumber (or PyMuPDF as fallback).
  4. Remove headers, footers, and the references section.
  5. Save a JSON sidecar per paper to data/processed/.

Design choices
--------------
* We store raw full-text + per-page text so that any chunker can work from the
  same artefact without re-parsing PDFs.
* Metadata is extracted heuristically (title from first page, authors from
  arXiv API response).  We fall back to filename when unavailable.
* The `clean_text` step uses heuristics rather than a trained model to keep
  the dependency footprint small.

Usage
-----
    python ingest.py                        # download 35 default papers
    python ingest.py --n 10 --query "BERT"  # custom count & query
    python ingest.py --pdf path/to/local.pdf  # parse a local PDF
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import arxiv
import pdfplumber
import requests
from tqdm import tqdm

from utils import ensure_dirs, fix_seed, get_config, get_logger

logger = get_logger(__name__)
cfg = get_config()

# ── Data Model ────────────────────────────────────────────────────────────────

@dataclass
class PaperMetadata:
    """Structured metadata for a single paper."""

    arxiv_id:   str
    title:      str
    authors:    list[str]
    year:       int
    abstract:   str
    url:        str
    pdf_path:   str                    # relative to project root
    sections:   list[str] = field(default_factory=list)   # detected section names
    page_count: int = 0


@dataclass
class ProcessedPaper:
    """Full extracted content ready for chunking."""

    metadata:      PaperMetadata
    full_text:     str                 # entire body, cleaned
    pages:         list[str]          # per-page text (cleaned)
    section_texts: dict[str, str]     # section_name → text


# ── arXiv Download ────────────────────────────────────────────────────────────

# Curated set of landmark deep-learning arXiv IDs to guarantee relevance.
DEFAULT_ARXIV_IDS: list[str] = [
    # Foundational architectures
    "1706.03762",  # Attention Is All You Need (Transformer)
    "1810.04805",  # BERT
    "2005.14165",  # GPT-3
    "2010.11929",  # Vision Transformer (ViT)
    "1512.03385",  # ResNet (He et al.)
    "1409.1556",   # VGGNet
    "1312.6114",   # VAE (Kingma & Welling)
    "1406.2661",   # GAN (Goodfellow et al.)
    # Training techniques
    "1502.03167",  # Batch Normalization
    "1207.0580",   # Dropout
    "1412.6980",   # Adam optimizer
    "1608.05343",  # Curriculum Learning (Bengio)
    # NLP advances
    "1910.10683",  # T5
    "1909.11942",  # RoBERTa
    "1910.01108",  # DistilBERT
    "2106.09685",  # LoRA
    "2302.13971",  # LLaMA
    "2307.09288",  # LLaMA 2
    # Vision / multimodal
    "2103.00020",  # CLIP
    "2204.02311",  # DALL·E 2
    "2112.10752",  # Latent Diffusion (Stable Diffusion)
    "2010.09700",  # DeiT
    "2111.11418",  # MAE (Masked Autoencoders)
    # Reinforcement / alignment
    "2203.02155",  # InstructGPT / RLHF
    "1707.06347",  # PPO
    # Graph / misc
    "1609.02907",  # GCN (Kipf & Welling)
    "1710.10903",  # GAT
    "2010.14701",  # DALL·E 1
    # Efficiency / scaling
    "2001.08361",  # Scaling Laws (Kaplan et al.)
    "2203.15556",  # Chinchilla
    "2205.01068",  # FlashAttention
    "2209.01188",  # LM-BFF / PEFT overview
    "2305.10403",  # QLoRA
    "1803.05457",  # ELMo
    "2104.09864",  # RoFormer (Rotary embeddings)
]


def fetch_arxiv_papers(
    arxiv_ids: list[str],
    papers_dir: Path,
    max_papers: int = 35,
) -> list[PaperMetadata]:
    """
    Download PDFs from arXiv and return metadata for each.

    Parameters
    ----------
    arxiv_ids   : list of arXiv IDs to fetch
    papers_dir  : directory to save PDFs
    max_papers  : cap on how many papers to download

    Returns
    -------
    list of PaperMetadata (one per successfully downloaded paper)
    """
    papers_dir.mkdir(parents=True, exist_ok=True)
    ids_to_fetch = arxiv_ids[:max_papers]
    logger.info("Fetching metadata for %d arXiv papers …", len(ids_to_fetch))

    client = arxiv.Client(page_size=50, delay_seconds=1.0, num_retries=3)
    search = arxiv.Search(id_list=ids_to_fetch)

    results: list[PaperMetadata] = []
    for result in tqdm(client.results(search), total=len(ids_to_fetch), desc="Downloading"):
        arxiv_id = result.entry_id.split("/")[-1]
        pdf_filename = f"{arxiv_id.replace('/', '_')}.pdf"
        pdf_path = papers_dir / pdf_filename

        if pdf_path.exists():
            logger.debug("Skipping %s (already downloaded)", arxiv_id)
        else:
            try:
                pdf_url = result.pdf_url
                if not pdf_url:
                    logger.warning("No PDF URL for %s", arxiv_id)
                    continue
                resp = requests.get(pdf_url, timeout=60)
                resp.raise_for_status()
                with open(pdf_path, "wb") as f_pdf:
                    f_pdf.write(resp.content)
                time.sleep(1.0)  # be polite to arXiv servers
            except Exception as exc:
                logger.warning("Failed to download %s: %s", arxiv_id, exc)
                continue

        meta = PaperMetadata(
            arxiv_id=arxiv_id,
            title=result.title,
            authors=[str(a) for a in result.authors],
            year=result.published.year,
            abstract=result.summary,
            url=result.entry_id,
            pdf_path=str(pdf_path.relative_to(Path(__file__).parent)),
        )
        results.append(meta)
        logger.info("[OK] %s — %s", arxiv_id, result.title[:60])

    logger.info("Downloaded %d papers.", len(results))
    return results


# ── PDF Parsing ───────────────────────────────────────────────────────────────

def _clean_line(line: str) -> str:
    """Remove common artifacts: page numbers, URLs, excessive whitespace."""
    line = line.strip()
    # Drop lines that are just page numbers (e.g. "— 3 —", "3", "Page 3")
    if re.fullmatch(r"[-–—\s]*\d+[-–—\s]*", line):
        return ""
    # Drop lines that look like running headers/footers (short, all caps, no spaces)
    if len(line) < 60 and line.isupper() and " " not in line:
        return ""
    return line


def _remove_references_section(text: str) -> str:
    """
    Strip everything from the References/Bibliography heading onwards.

    Uses a regex that matches common heading styles.
    """
    pattern = re.compile(
        r"\n\s*(?:\d+\.)?\s*(?:references|bibliography|works cited)\s*\n",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if match:
        return text[: match.start()]
    return text


def _detect_sections(text: str, patterns: list[str]) -> dict[str, str]:
    """
    Split text into sections based on header regex patterns.

    Returns
    -------
    dict mapping section_name → section_text
    """
    compiled = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in patterns]
    lines = text.split("\n")

    sections: dict[str, list[str]] = {"preamble": []}
    current = "preamble"

    for line in lines:
        matched = False
        for pat in compiled:
            if pat.match(line.strip()):
                current = line.strip().lower()
                sections[current] = []
                matched = True
                break
        if not matched:
            sections[current].append(line)

    return {k: "\n".join(v).strip() for k, v in sections.items() if v}


def parse_pdf_pdfplumber(
    pdf_path: Path,
    remove_refs: bool = True,
    remove_headers: bool = True,
) -> tuple[str, list[str]]:
    """
    Extract text from a PDF using pdfplumber.

    Returns
    -------
    (full_text, pages)  where pages is a list of per-page cleaned text
    """
    pages_text: list[str] = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                raw = page.extract_text() or ""
                if remove_headers:
                    cleaned_lines = [_clean_line(ln) for ln in raw.split("\n")]
                    raw = "\n".join(ln for ln in cleaned_lines if ln)
                pages_text.append(raw)
    except Exception as exc:
        logger.warning("pdfplumber failed on %s: %s — trying PyMuPDF", pdf_path.name, exc)
        return parse_pdf_pymupdf(pdf_path, remove_refs, remove_headers)

    full = "\n\n".join(pages_text)
    if remove_refs:
        full = _remove_references_section(full)
    return full, pages_text


def parse_pdf_pymupdf(
    pdf_path: Path,
    remove_refs: bool = True,
    remove_headers: bool = True,
) -> tuple[str, list[str]]:
    """
    Fallback PDF parser using PyMuPDF (fitz).

    Returns
    -------
    (full_text, pages)
    """
    import fitz  # PyMuPDF

    pages_text: list[str] = []
    doc = fitz.open(str(pdf_path))
    for page in doc:
        raw = page.get_text("text")
        if remove_headers:
            cleaned_lines = [_clean_line(ln) for ln in raw.split("\n")]
            raw = "\n".join(ln for ln in cleaned_lines if ln)
        pages_text.append(raw)
    doc.close()

    full = "\n\n".join(pages_text)
    if remove_refs:
        full = _remove_references_section(full)
    return full, pages_text


def parse_pdf(pdf_path: Path) -> tuple[str, list[str]]:
    """Dispatch to the parser configured in config.yaml."""
    lib = cfg["pdf"].get("library", "pdfplumber")
    remove_refs = cfg["pdf"].get("remove_references", True)
    remove_hdr = cfg["pdf"].get("remove_headers_footers", True)

    if lib == "pymupdf":
        return parse_pdf_pymupdf(pdf_path, remove_refs, remove_hdr)
    return parse_pdf_pdfplumber(pdf_path, remove_refs, remove_hdr)


# ── Full Processing Pipeline ──────────────────────────────────────────────────

def process_paper(meta: PaperMetadata) -> ProcessedPaper | None:
    """
    Parse a single PDF and produce a ProcessedPaper.

    Parameters
    ----------
    meta : PaperMetadata (must have a valid pdf_path)

    Returns
    -------
    ProcessedPaper or None on failure
    """
    pdf_path = Path(__file__).parent / meta.pdf_path
    if not pdf_path.exists():
        logger.warning("PDF not found: %s", pdf_path)
        return None

    logger.info("Parsing %s …", pdf_path.name)
    try:
        full_text, pages = parse_pdf(pdf_path)
    except Exception as exc:
        logger.error("Failed to parse %s: %s", pdf_path.name, exc)
        return None

    section_patterns = cfg["chunking"]["section"]["header_patterns"]
    section_texts = _detect_sections(full_text, section_patterns)
    meta.sections = list(section_texts.keys())
    meta.page_count = len(pages)

    return ProcessedPaper(
        metadata=meta,
        full_text=full_text,
        pages=pages,
        section_texts=section_texts,
    )


def save_processed(paper: ProcessedPaper, processed_dir: Path) -> Path:
    """
    Serialize a ProcessedPaper to JSON.

    Returns
    -------
    Path to the saved JSON file
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    safe_id = paper.metadata.arxiv_id.replace("/", "_")
    out_path = processed_dir / f"{safe_id}.json"

    payload = {
        "metadata": asdict(paper.metadata),
        "full_text": paper.full_text,
        "pages": paper.pages,
        "section_texts": paper.section_texts,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.debug("Saved processed paper → %s", out_path.name)
    return out_path


def load_processed(processed_dir: Path) -> list[ProcessedPaper]:
    """
    Load all processed papers from disk.

    Returns
    -------
    list of ProcessedPaper objects
    """
    papers = []
    for json_file in sorted(processed_dir.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        meta = PaperMetadata(**data["metadata"])
        paper = ProcessedPaper(
            metadata=meta,
            full_text=data["full_text"],
            pages=data["pages"],
            section_texts=data["section_texts"],
        )
        papers.append(paper)
    return papers


def ingest_local_pdf(pdf_path: str | Path) -> ProcessedPaper | None:
    """
    Parse a user-supplied local PDF (used by the Streamlit UI upload feature).

    Parameters
    ----------
    pdf_path : path to the PDF file

    Returns
    -------
    ProcessedPaper or None on failure
    """
    pdf_path = Path(pdf_path)
    meta = PaperMetadata(
        arxiv_id=pdf_path.stem,
        title=pdf_path.stem.replace("_", " ").title(),
        authors=[],
        year=0,
        abstract="",
        url="",
        pdf_path=str(pdf_path),
    )
    return process_paper(meta)


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest arXiv papers into the RAG knowledge base")
    parser.add_argument("--n", type=int, default=35, help="Number of papers to download")
    parser.add_argument("--query", type=str, default=None, help="Custom arXiv search query")
    parser.add_argument("--pdf", type=str, default=None, help="Path to a local PDF to ingest")
    args = parser.parse_args()

    fix_seed()
    ensure_dirs()

    root = Path(__file__).parent
    papers_dir = root / cfg["paths"]["papers_dir"]
    processed_dir = root / cfg["paths"]["processed_dir"]

    if args.pdf:
        # Single local PDF mode
        paper = ingest_local_pdf(args.pdf)
        if paper:
            save_processed(paper, processed_dir)
            logger.info("Local PDF ingested → %s", args.pdf)
        return

    # arXiv bulk download mode
    if args.query:
        logger.info("Searching arXiv for: %s", args.query)
        client = arxiv.Client()
        search = arxiv.Search(
            query=args.query,
            max_results=args.n,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        ids = [r.entry_id.split("/")[-1] for r in client.results(search)]
    else:
        ids = DEFAULT_ARXIV_IDS

    metas = fetch_arxiv_papers(ids, papers_dir, max_papers=args.n)

    succeeded = 0
    for meta in tqdm(metas, desc="Processing PDFs"):
        paper = process_paper(meta)
        if paper:
            save_processed(paper, processed_dir)
            succeeded += 1

    logger.info("Ingestion complete: %d / %d papers processed.", succeeded, len(metas))


if __name__ == "__main__":
    main()
