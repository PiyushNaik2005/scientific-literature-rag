"""
chunking.py — Three text-chunking strategies with configurable size and overlap.

Strategies
----------
1. FixedChunker      — splits text into fixed-character windows with overlap.
   Pros: simple, predictable size; Cons: breaks mid-sentence.

2. RecursiveChunker  — tries longer separators first (\n\n, \n, ". ", " "), so
   chunks respect paragraph/sentence boundaries whenever possible.
   Pros: more coherent chunks; Cons: variable chunk size.

3. SectionChunker    — splits on detected section headings, then sub-splits
   large sections with the recursive strategy.
   Pros: preserves section context in metadata; Cons: depends on heading quality.

Each chunker produces a list of `Chunk` objects that carry both the text and
rich metadata so the retriever can display proper citations.

Usage
-----
    from chunking import chunk_paper, get_chunker
    from ingest import load_processed

    papers = load_processed(Path("data/processed"))
    chunks  = chunk_paper(papers[0], strategy="recursive")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ingest import ProcessedPaper
from utils import get_config, get_logger

logger = get_logger(__name__)
cfg = get_config()

# ── Data Model ────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """A single text chunk with provenance metadata."""

    chunk_id:    str          # unique: "<arxiv_id>_<strategy>_<index>"
    text:        str          # chunk body
    arxiv_id:    str
    title:       str
    authors:     list[str]
    year:        int
    url:         str
    strategy:    str          # "fixed" | "recursive" | "section"
    section:     str = ""     # section name (if available)
    page_hint:   int = -1     # approximate source page (-1 if unknown)
    chunk_index: int = 0      # position within this paper's chunks

    def citation(self) -> str:
        """Human-readable citation string for UI display."""
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."
        section_str = f" [{self.section}]" if self.section else ""
        return f"{self.title}{section_str} — {authors_str} ({self.year})"


# ── Base Chunker ──────────────────────────────────────────────────────────────

class BaseChunker:
    """Abstract base; subclasses implement `split_text`."""

    strategy: str = "base"

    def split_text(self, text: str) -> list[str]:  # pragma: no cover
        raise NotImplementedError

    def chunk_paper(self, paper: ProcessedPaper) -> list[Chunk]:
        """Chunk a full paper and attach metadata to every chunk."""
        segments = self.split_text(paper.full_text)
        chunks = []
        for idx, seg in enumerate(segments):
            seg = seg.strip()
            if not seg:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{paper.metadata.arxiv_id}_{self.strategy}_{idx:04d}",
                    text=seg,
                    arxiv_id=paper.metadata.arxiv_id,
                    title=paper.metadata.title,
                    authors=paper.metadata.authors,
                    year=paper.metadata.year,
                    url=paper.metadata.url,
                    strategy=self.strategy,
                    chunk_index=idx,
                )
            )
        logger.debug(
            "[%s] %s → %d chunks (strategy=%s)",
            self.__class__.__name__,
            paper.metadata.arxiv_id,
            len(chunks),
            self.strategy,
        )
        return chunks


# ── 1. Fixed-Size Chunker ─────────────────────────────────────────────────────

class FixedChunker(BaseChunker):
    """
    Split text into fixed-character windows with configurable overlap.

    Parameters
    ----------
    chunk_size : window size in characters
    overlap    : number of characters shared between consecutive chunks
    """

    strategy = "fixed"

    def __init__(
        self,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> None:
        fixed_cfg = cfg["chunking"]["fixed"]
        self.chunk_size = chunk_size or fixed_cfg["chunk_size"]
        self.overlap    = overlap    or fixed_cfg["overlap"]

    def split_text(self, text: str) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.overlap
        return chunks


# ── 2. Recursive / Sentence-Based Chunker ────────────────────────────────────

class RecursiveChunker(BaseChunker):
    """
    Recursively split text by a priority list of separators.

    This mimics LangChain's RecursiveCharacterTextSplitter without the
    dependency.  The algorithm tries the first separator; if a resulting
    piece is still larger than `chunk_size`, it recurses with the next
    separator in the list.

    Parameters
    ----------
    chunk_size  : target chunk size in characters
    overlap     : character overlap between chunks
    separators  : ordered list of split tokens (most preferred first)
    """

    strategy = "recursive"

    def __init__(
        self,
        chunk_size:  int | None = None,
        overlap:     int | None = None,
        separators:  list[str] | None = None,
    ) -> None:
        rec_cfg = cfg["chunking"]["recursive"]
        self.chunk_size  = chunk_size  or rec_cfg["chunk_size"]
        self.overlap     = overlap     or rec_cfg["overlap"]
        self.separators  = separators  or rec_cfg["separators"]

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            # Base case: no more separators; split by chunk_size
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size - self.overlap)
            ]

        sep = separators[0]
        rest = separators[1:]
        pieces = text.split(sep) if sep else list(text)

        merged: list[str] = []
        current = ""

        for piece in pieces:
            candidate = (current + sep + piece) if current else piece
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                # current is full — flush it
                if current:
                    if len(current) > self.chunk_size:
                        # Still too big — recurse with next separator
                        merged.extend(self._split(current, rest))
                    else:
                        merged.append(current)
                current = piece

        if current:
            if len(current) > self.chunk_size:
                merged.extend(self._split(current, rest))
            else:
                merged.append(current)

        # Apply overlap by blending adjacent chunks
        return self._apply_overlap(merged)

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        if self.overlap <= 0 or len(chunks) <= 1:
            return chunks
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = result[-1][-self.overlap:]
            result.append(tail + chunks[i])
        return result

    def split_text(self, text: str) -> list[str]:
        return self._split(text, self.separators)


# ── 3. Section-Aware Chunker ──────────────────────────────────────────────────

class SectionChunker(BaseChunker):
    """
    Split text on detected section headings, then sub-split large sections.

    Each chunk carries the section name in its metadata so the retriever
    can display "Results section of paper X" in citations.

    Parameters
    ----------
    header_patterns         : list of regex strings matching section headings
    max_section_chunk_size  : sections larger than this are sub-split
    overlap                 : overlap for the sub-split step
    """

    strategy = "section"

    def __init__(
        self,
        header_patterns:        list[str] | None = None,
        max_section_chunk_size: int | None = None,
        overlap:                int | None = None,
    ) -> None:
        sec_cfg = cfg["chunking"]["section"]
        self.header_patterns        = header_patterns        or sec_cfg["header_patterns"]
        self.max_section_chunk_size = max_section_chunk_size or sec_cfg["max_section_chunk_size"]
        self.overlap                = overlap                or sec_cfg["overlap"]
        self._sub_chunker           = RecursiveChunker(
            chunk_size=self.max_section_chunk_size,
            overlap=self.overlap,
        )
        self._compiled = [
            re.compile(p, re.IGNORECASE | re.MULTILINE)
            for p in self.header_patterns
        ]

    def _split_into_sections(self, text: str) -> list[tuple[str, str]]:
        """Return list of (section_name, section_text) tuples."""
        lines = text.split("\n")
        sections: list[tuple[str, list[str]]] = [("preamble", [])]

        for line in lines:
            stripped = line.strip()
            matched = False
            for pat in self._compiled:
                if pat.match(stripped):
                    sections.append((stripped.lower()[:60], []))
                    matched = True
                    break
            if not matched:
                sections[-1][1].append(line)

        return [(name, "\n".join(lines_).strip()) for name, lines_ in sections if lines_]

    def split_text(self, text: str) -> list[str]:
        """Override to produce plain list; section metadata added in chunk_paper."""
        sections = self._split_into_sections(text)
        all_chunks = []
        for _, sec_text in sections:
            if len(sec_text) <= self.max_section_chunk_size:
                all_chunks.append(sec_text)
            else:
                all_chunks.extend(self._sub_chunker.split_text(sec_text))
        return all_chunks

    def chunk_paper(self, paper: ProcessedPaper) -> list[Chunk]:
        """Override to inject section names into chunk metadata."""
        sections = self._split_into_sections(paper.full_text)
        chunks: list[Chunk] = []
        global_idx = 0

        for sec_name, sec_text in sections:
            if not sec_text.strip():
                continue
            if len(sec_text) <= self.max_section_chunk_size:
                sub_texts = [sec_text]
            else:
                sub_texts = self._sub_chunker.split_text(sec_text)

            for seg in sub_texts:
                seg = seg.strip()
                if not seg:
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=f"{paper.metadata.arxiv_id}_section_{global_idx:04d}",
                        text=seg,
                        arxiv_id=paper.metadata.arxiv_id,
                        title=paper.metadata.title,
                        authors=paper.metadata.authors,
                        year=paper.metadata.year,
                        url=paper.metadata.url,
                        strategy="section",
                        section=sec_name,
                        chunk_index=global_idx,
                    )
                )
                global_idx += 1

        logger.debug(
            "[SectionChunker] %s → %d chunks across %d sections",
            paper.metadata.arxiv_id,
            len(chunks),
            len(sections),
        )
        return chunks


# ── Factory ───────────────────────────────────────────────────────────────────

CHUNKER_MAP: dict[str, type[BaseChunker]] = {
    "fixed":     FixedChunker,
    "recursive": RecursiveChunker,
    "section":   SectionChunker,
}


def get_chunker(strategy: str | None = None, **kwargs) -> BaseChunker:
    """
    Return a chunker instance for the given strategy.

    Parameters
    ----------
    strategy : "fixed" | "recursive" | "section" (defaults to config value)
    **kwargs : passed through to the chunker constructor

    Returns
    -------
    An initialised BaseChunker subclass
    """
    if strategy is None:
        strategy = cfg["chunking"]["default_strategy"]
    if strategy not in CHUNKER_MAP:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {list(CHUNKER_MAP)}")
    return CHUNKER_MAP[strategy](**kwargs)


def chunk_paper(
    paper: ProcessedPaper,
    strategy: str | None = None,
    **kwargs,
) -> list[Chunk]:
    """
    Convenience function: chunk a single paper with the given strategy.

    Parameters
    ----------
    paper    : ProcessedPaper from ingest.py
    strategy : "fixed" | "recursive" | "section" (defaults to config)
    **kwargs : forwarded to the chunker constructor

    Returns
    -------
    list of Chunk
    """
    chunker = get_chunker(strategy, **kwargs)
    return chunker.chunk_paper(paper)


def chunk_all_papers(
    papers: list[ProcessedPaper],
    strategy: str | None = None,
    **kwargs,
) -> list[Chunk]:
    """
    Chunk every paper and return a flat list of all chunks.

    Parameters
    ----------
    papers   : list of ProcessedPaper
    strategy : chunking strategy
    **kwargs : forwarded to chunker

    Returns
    -------
    flat list of Chunk objects
    """
    all_chunks: list[Chunk] = []
    for paper in papers:
        all_chunks.extend(chunk_paper(paper, strategy, **kwargs))
    logger.info(
        "Chunked %d papers → %d total chunks (strategy=%s)",
        len(papers),
        len(all_chunks),
        strategy or cfg["chunking"]["default_strategy"],
    )
    return all_chunks
