"""
generator.py — LLM answer generation with citation injection and
               hallucination detection.

Architecture
------------
  Retrieved chunks ──►  prompt builder  ──►  LLM  ──►  raw answer
                                                            │
                                               hallucination checker
                                                            │
                                             GenerationResult (answer +
                                              citations + flagged sentences)

LLM Providers (all behind one interface)
-----------------------------------------
  openai      → OpenAI GPT-4o via openai SDK
  gemini      → Google Gemini via google-generativeai SDK
  huggingface → Local model via transformers pipeline

Design choices
--------------
* The system prompt strictly constrains the model to answer *only* from the
  provided context and to include [CITATION: chunk_id] markers inline.
* A post-generation NLI pass (DeBERTa cross-encoder) checks each sentence
  against the retrieved passages and flags those with low entailment score.
* When no chunks pass the similarity threshold, we return a canned "not found"
  message instead of querying the LLM, avoiding hallucinations at zero cost.

Usage
-----
    from generator import Generator
    from retriever import RetrievalResult

    gen = Generator()
    result = gen.generate(query="What is LoRA?", results=retrieved_results)
    print(result.answer)
    print(result.citations)
    print(result.flagged_sentences)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

from retriever import RetrievalResult
from utils import get_config, get_logger

load_dotenv()  # load API keys from .env
logger = get_logger(__name__)
cfg = get_config()

# ── Response model ────────────────────────────────────────────────────────────

@dataclass
class Citation:
    """A single citation linked to a sentence in the answer."""
    chunk_id:  str
    title:     str
    authors:   list[str]
    year:      int
    section:   str
    url:       str
    excerpt:   str    # first 120 chars of the chunk

    def display(self) -> str:
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."
        section_str = f" [{self.section}]" if self.section else ""
        return f"**{self.title}**{section_str} — {authors_str} ({self.year})"


@dataclass
class GenerationResult:
    """Complete answer with citations and hallucination flags."""
    query:              str
    answer:             str
    citations:          list[Citation]
    flagged_sentences:  list[dict]     # [{"sentence": str, "score": float}]
    not_found:          bool = False   # True when no context passed threshold
    raw_llm_response:   str = ""


# ── Prompt engineering ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise scientific literature assistant. Your task is to answer the user's question using ONLY the provided context passages.

Rules you MUST follow:
1. Base every claim strictly on the provided context. Do not add outside knowledge.
2. After each claim or sentence, cite the source by writing [CITATION: chunk_id] using the exact chunk_id shown in the context.
3. If the answer cannot be found in the provided passages, respond with exactly: "NOT_FOUND_IN_PROVIDED_PAPERS"
4. Do not speculate, guess, or extrapolate beyond what is written.
5. Be concise but thorough. Use scientific language appropriate to the domain.
6. If multiple passages support a claim, cite all of them.

Format your response as flowing prose with inline citations. Do not use bullet points unless the question explicitly asks for a list."""

CONTEXT_TEMPLATE = """
--- CONTEXT PASSAGE {rank} ---
Chunk ID: {chunk_id}
Source: {title} ({year})
Authors: {authors}
Section: {section}
Similarity score: {score:.3f}

{text}
---
"""

NOT_FOUND_RESPONSE = (
    "I could not find information about this in the provided papers. "
    "The retrieved passages do not contain sufficient relevant content to answer this question reliably."
)


def build_prompt(query: str, results: list[RetrievalResult]) -> str:
    """
    Build a context-grounded prompt from retrieved passages.

    Parameters
    ----------
    query   : user question
    results : list of RetrievalResult (already filtered and re-ranked)

    Returns
    -------
    Full prompt string (context + question)
    """
    context_blocks = []
    for r in results:
        authors_str = ", ".join(r.chunk.authors[:3])
        if len(r.chunk.authors) > 3:
            authors_str += " et al."
        block = CONTEXT_TEMPLATE.format(
            rank=r.rank,
            chunk_id=r.chunk.chunk_id,
            title=r.chunk.title,
            year=r.chunk.year,
            authors=authors_str,
            section=r.chunk.section or "N/A",
            score=r.vector_score,
            text=r.chunk.text.strip(),
        )
        context_blocks.append(block)

    context_str = "\n".join(context_blocks)
    return f"{context_str}\n\nQUESTION: {query}\n\nANSWER:"


# ── Citation extraction ───────────────────────────────────────────────────────

def extract_citations(
    answer: str,
    results: list[RetrievalResult],
) -> tuple[str, list[Citation]]:
    """
    Parse [CITATION: chunk_id] markers from the answer text and return
    a cleaned answer + structured Citation objects.

    Parameters
    ----------
    answer  : raw LLM output containing [CITATION: ...] markers
    results : retrieval results (used to look up chunk metadata)

    Returns
    -------
    (cleaned_answer, citations)
    """
    chunk_map = {r.chunk.chunk_id: r for r in results}
    seen_ids: set[str] = set()
    citations: list[Citation] = []

    pattern = re.compile(r"\[CITATION:\s*([^\]]+)\]")

    def replace_citation(match: re.Match) -> str:
        cid = match.group(1).strip()
        if cid in chunk_map and cid not in seen_ids:
            seen_ids.add(cid)
            r = chunk_map[cid]
            citations.append(
                Citation(
                    chunk_id=cid,
                    title=r.chunk.title,
                    authors=r.chunk.authors,
                    year=r.chunk.year,
                    section=r.chunk.section,
                    url=r.chunk.url,
                    excerpt=r.chunk.text[:120].replace("\n", " "),
                )
            )
            # Replace inline with superscript-style reference number
            return f"[{len(seen_ids)}]"
        return f"[?{cid}]"   # unknown chunk id

    cleaned = pattern.sub(replace_citation, answer)
    return cleaned.strip(), citations


# ── Hallucination detector ────────────────────────────────────────────────────

class HallucinationChecker:
    """
    Per-sentence entailment check: each sentence in the answer is scored
    against the concatenated retrieved passages using an NLI cross-encoder.

    A sentence is flagged if its entailment score is below the configured
    threshold.  This is approximate — NLI models have their own errors —
    but catches obvious hallucinations.
    """

    def __init__(self) -> None:
        if not cfg["hallucination"]["flag_unsupported"]:
            self._model = None
            return
        from sentence_transformers import CrossEncoder
        model_name = cfg["hallucination"]["entailment_model"]
        logger.info("Loading NLI model '%s' …", model_name)
        self._model = CrossEncoder(model_name, num_labels=3)
        # Label order: contradiction=0, neutral=1, entailment=2
        self._threshold = cfg["hallucination"]["entailment_threshold"]

    def check(
        self,
        answer: str,
        results: list[RetrievalResult],
    ) -> list[dict]:
        """
        Check each sentence of the answer against the retrieved context.

        Parameters
        ----------
        answer  : cleaned answer text
        results : retrieval results used as the premise

        Returns
        -------
        list of {"sentence": str, "score": float, "supported": bool}
        for sentences where supported=False (flagged only)
        """
        if self._model is None or not results:
            return []

        # Build a single premise from all retrieved passages
        premise = " ".join(r.chunk.text[:300] for r in results)

        # Split answer into sentences (simple heuristic)
        sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
        sentences = [s for s in sentences if len(s.split()) >= 5]  # skip very short

        flagged = []
        for sent in sentences:
            scores = self._model.predict([(premise, sent)])[0]
            # scores shape: (3,) → [contradiction, neutral, entailment]
            entailment_score = float(scores[2])
            if entailment_score < self._threshold:
                flagged.append(
                    {
                        "sentence": sent,
                        "score": entailment_score,
                        "supported": False,
                    }
                )
        return flagged


# ── LLM Provider implementations ──────────────────────────────────────────────

class OpenAIProvider:
    """OpenAI GPT-4o completion."""

    def __init__(self) -> None:
        import openai
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set. Add it to your .env file.")
        self._client = openai.OpenAI(api_key=api_key)
        self._model  = cfg["generation"]["openai"]["model"]
        self._temp   = cfg["generation"]["openai"]["temperature"]
        self._max_t  = cfg["generation"]["openai"]["max_tokens"]

    def complete(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=self._temp,
            max_tokens=self._max_t,
        )
        return response.choices[0].message.content or ""


class GeminiProvider:
    """Google Gemini completion."""

    def __init__(self) -> None:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set. Add it to your .env file.")
        genai.configure(api_key=api_key)
        gcfg = cfg["generation"]["gemini"]
        self._model = genai.GenerativeModel(
            model_name=gcfg["model"],
            system_instruction=SYSTEM_PROMPT,
            generation_config=genai.types.GenerationConfig(
                temperature=gcfg["temperature"],
                max_output_tokens=gcfg["max_tokens"],
            ),
        )

    def complete(self, prompt: str) -> str:
        response = self._model.generate_content(prompt)
        return response.text or ""


class HuggingFaceProvider:
    """Local Hugging Face model via transformers pipeline."""

    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        hf_cfg  = cfg["generation"]["huggingface"]
        model_name = hf_cfg["model"]
        logger.info("Loading local HF model '%s' …", model_name)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        load_kwargs: dict = {"device_map": "auto"}
        if hf_cfg.get("load_in_4bit") and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self._pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=hf_cfg["max_new_tokens"],
            temperature=hf_cfg["temperature"],
            do_sample=True,
        )
        self._sys = SYSTEM_PROMPT

    def complete(self, prompt: str) -> str:
        full_prompt = f"<|system|>\n{self._sys}\n<|user|>\n{prompt}\n<|assistant|>\n"
        output = self._pipe(full_prompt)[0]["generated_text"]
        # Strip the prompt prefix
        if "<|assistant|>" in output:
            output = output.split("<|assistant|>")[-1].strip()
        return output


class ExtractiveProvider:
    """
    Extractive synthesis provider that generates answers directly from
    the retrieved context passages with exact inline citations.
    Works offline with zero API keys and zero heavy model downloads.
    """

    def __init__(self) -> None:
        logger.info("Initialising Extractive synthesis provider.")

    def complete(self, prompt: str) -> str:
        q_match = re.search(r"QUESTION:\s*(.+?)\n\nANSWER:", prompt, re.DOTALL)
        query = q_match.group(1).strip() if q_match else ""

        # Extract passages:
        passages = re.findall(
            r"--- CONTEXT PASSAGE \d+ ---\s*Chunk ID:\s*(\S+).*?\n\n(.*?)\n---",
            prompt,
            re.DOTALL,
        )
        if not passages:
            return "NOT_FOUND_IN_PROVIDED_PAPERS"

        q_tokens = set(re.findall(r"\w{3,}", query.lower()))
        assembled_claims = []

        for cid, text in passages[:4]:
            text_cleaned = re.sub(r"\s+", " ", text).strip()
            sents = re.split(r"(?<=[.!?])\s+", text_cleaned)
            best_sent = None
            best_score = -1

            for s in sents:
                s_strip = s.strip()
                if len(s_strip.split()) < 6 or len(s_strip.split()) > 50:
                    continue
                s_words = set(re.findall(r"\w{3,}", s_strip.lower()))
                overlap = len(s_words & q_tokens)
                if overlap > best_score:
                    best_score = overlap
                    best_sent = s_strip

            if not best_sent and sents:
                for s in sents:
                    if len(s.split()) >= 6:
                        best_sent = s.strip()
                        break

            if best_sent:
                if not best_sent.endswith((".", "!", "?")):
                    best_sent += "."
                assembled_claims.append(f"{best_sent} [CITATION: {cid}]")

        if not assembled_claims:
            return "NOT_FOUND_IN_PROVIDED_PAPERS"

        return " ".join(assembled_claims)


_PROVIDER_MAP = {
    "openai":       OpenAIProvider,
    "gemini":       GeminiProvider,
    "huggingface":  HuggingFaceProvider,
    "extractive":   ExtractiveProvider,
}


# ── Main Generator ────────────────────────────────────────────────────────────

class Generator:
    """
    Orchestrates prompt construction, LLM generation, citation extraction,
    and hallucination detection.

    Parameters
    ----------
    provider : "openai" | "gemini" | "huggingface" | "extractive" (defaults to config)
    """

    def __init__(self, provider: str | None = None) -> None:
        provider = provider or cfg["generation"]["llm_provider"]
        if provider not in _PROVIDER_MAP:
            logger.warning("Unknown provider '%s', defaulting to 'extractive'", provider)
            provider = "extractive"
        logger.info("Initialising Generator with provider='%s'", provider)
        try:
            self._llm = _PROVIDER_MAP[provider]()
        except Exception as exc:
            logger.warning(
                "Could not initialize provider '%s' (%s). Falling back to 'extractive' provider.",
                provider,
                exc,
            )
            self._llm = ExtractiveProvider()
        self._checker = HallucinationChecker()

    def generate(
        self,
        query:   str,
        results: list[RetrievalResult],
    ) -> GenerationResult:
        """
        Generate a grounded answer for the given query.

        Parameters
        ----------
        query   : user question
        results : retrieved and re-ranked chunks (may be empty)

        Returns
        -------
        GenerationResult with answer, citations, and flagged sentences
        """
        # 1. Handle empty / below-threshold retrieval
        if not results:
            logger.info("No results above threshold → returning NOT_FOUND response.")
            return GenerationResult(
                query=query,
                answer=NOT_FOUND_RESPONSE,
                citations=[],
                flagged_sentences=[],
                not_found=True,
            )

        # 2. Build prompt
        prompt = build_prompt(query, results)

        # 3. Call LLM
        logger.info("Calling LLM for query: %s", query[:80])
        raw = self._llm.complete(prompt)
        logger.debug("Raw LLM output (%d chars): %s …", len(raw), raw[:200])

        # 4. Check for explicit NOT_FOUND signal
        if "NOT_FOUND_IN_PROVIDED_PAPERS" in raw:
            return GenerationResult(
                query=query,
                answer=NOT_FOUND_RESPONSE,
                citations=[],
                flagged_sentences=[],
                not_found=True,
                raw_llm_response=raw,
            )

        # 5. Extract citations
        cleaned_answer, citations = extract_citations(raw, results)

        # 6. Hallucination check
        flagged = self._checker.check(cleaned_answer, results)
        if flagged:
            logger.warning(
                "%d sentences flagged as potentially unsupported.", len(flagged)
            )

        return GenerationResult(
            query=query,
            answer=cleaned_answer,
            citations=citations,
            flagged_sentences=flagged,
            not_found=False,
            raw_llm_response=raw,
        )
