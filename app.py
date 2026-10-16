"""
app.py — Streamlit UI for the Scientific Literature RAG system.

Pages
-----
1. 💬 Ask a Question  — main Q&A interface with citations and hallucination flags
2. 📚 Knowledge Base  — browse papers and upload new PDFs
3. ⚙️  Settings        — tweak embedding model, chunking strategy, top-k live
4. 📊 Evaluation      — run evaluations and view results dashboard

Usage
-----
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ── Path bootstrap ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
load_dotenv()

from chunking import chunk_all_papers, chunk_paper, get_chunker
from embeddings import get_embedding_model
from generator import Generator, GenerationResult
from ingest import ingest_local_pdf, load_processed, save_processed
from retriever import Retriever, build_retriever
from utils import ensure_dirs, fix_seed, get_config

cfg = get_config()
fix_seed()
ensure_dirs()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SciRAG — Scientific Literature Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS styling ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-primary:    #0d1117;
        --bg-secondary:  #161b22;
        --bg-tertiary:   #21262d;
        --accent-blue:   #58a6ff;
        --accent-green:  #3fb950;
        --accent-purple: #bc8cff;
        --accent-orange: #f78166;
        --text-primary:  #e6edf3;
        --text-muted:    #8b949e;
        --border-color:  #30363d;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: var(--bg-primary);
        color: var(--text-primary);
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid var(--border-color);
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.95rem;
        padding: 6px 0;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 18px 24px;
        margin: 8px 0;
        transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: var(--accent-blue); }
    .metric-card .label { color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; }
    .metric-card .value { font-size: 2rem; font-weight: 700; color: var(--accent-blue); margin-top: 4px; }

    /* ── Citation block ── */
    .citation-block {
        background: linear-gradient(135deg, #1c2128, #21262d);
        border-left: 3px solid var(--accent-blue);
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 6px 0;
        font-size: 0.88rem;
    }
    .citation-block .cite-title { font-weight: 600; color: var(--accent-blue); }
    .citation-block .cite-meta  { color: var(--text-muted); margin-top: 4px; }
    .citation-block .cite-excerpt { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: var(--text-muted); margin-top: 8px; border-left: 2px solid var(--border-color); padding-left: 8px; }

    /* ── Flag block ── */
    .flag-block {
        background: linear-gradient(135deg, #1c1a0e, #21200d);
        border-left: 3px solid var(--accent-orange);
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        margin: 4px 0;
        font-size: 0.85rem;
    }
    .flag-block .flag-sentence { color: #e6b450; }
    .flag-block .flag-score    { color: var(--text-muted); font-size: 0.78rem; }

    /* ── Answer box ── */
    .answer-box {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 24px 28px;
        line-height: 1.75;
        font-size: 0.96rem;
    }

    /* ── Paper card ── */
    .paper-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
        transition: all 0.2s;
    }
    .paper-card:hover { border-color: var(--accent-blue); transform: translateY(-1px); }
    .paper-title { font-weight: 600; font-size: 0.95rem; color: var(--text-primary); }
    .paper-meta  { color: var(--text-muted); font-size: 0.82rem; margin-top: 4px; }

    /* ── Status badges ── */
    .badge-green  { background: #1a3226; color: var(--accent-green);  border-radius: 20px; padding: 2px 10px; font-size: 0.75rem; font-weight: 600; }
    .badge-orange { background: #2d1f0a; color: var(--accent-orange); border-radius: 20px; padding: 2px 10px; font-size: 0.75rem; font-weight: 600; }
    .badge-blue   { background: #0d2137; color: var(--accent-blue);   border-radius: 20px; padding: 2px 10px; font-size: 0.75rem; font-weight: 600; }

    /* ── Divider ── */
    hr { border-color: var(--border-color); }

    /* ── Input overrides ── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
        border-radius: 8px !important;
    }

    /* ── Button overrides ── */
    .stButton > button {
        background: linear-gradient(135deg, #1f6feb, #388bfd);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 20px;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state defaults ────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "embedding_model":   cfg["embeddings"]["default_model"],
        "chunking_strategy": cfg["chunking"]["default_strategy"],
        "top_k":             cfg["retrieval"]["final_k"],
        "use_reranker":      cfg["retrieval"]["use_reranker"],
        "threshold":         cfg["retrieval"]["similarity_threshold"],
        "retriever":         None,
        "generator":         None,
        "papers":            None,
        "chat_history":      [],
        "page":              "💬 Ask a Question",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading papers …")
def load_papers():
    processed_dir = ROOT / cfg["paths"]["processed_dir"]
    return load_processed(processed_dir)


@st.cache_resource(
    show_spinner="Building index …",
    hash_funcs={str: lambda x: x},
)
def get_retriever(model_slug: str, strategy: str):
    papers = load_papers()
    if not papers:
        return None
    chunks = chunk_all_papers(papers, strategy=strategy)
    return build_retriever(chunks, embedding_model_slug=model_slug, chunking_strategy=strategy)


@st.cache_resource(show_spinner="Loading generator …")
def get_generator():
    try:
        return Generator()
    except Exception as e:
        st.error(f"⚠️ Generator init failed: {e}")
        return None


# ── Sidebar navigation ────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔬 SciRAG")
    st.markdown("<small style='color:#8b949e'>Scientific Literature Assistant</small>", unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["💬 Ask a Question", "📚 Knowledge Base", "⚙️ Settings", "📊 Evaluation"],
        index=["💬 Ask a Question", "📚 Knowledge Base", "⚙️ Settings", "📊 Evaluation"].index(
            st.session_state.page
        ),
        label_visibility="collapsed",
    )
    st.session_state.page = page

    st.markdown("---")
    # Quick stats
    papers = load_papers()
    n_papers = len(papers) if papers else 0
    retriever = get_retriever(st.session_state.embedding_model, st.session_state.chunking_strategy)
    n_chunks  = retriever.collection_size() if retriever else 0

    st.markdown(f"**📄 Papers:** {n_papers}")
    st.markdown(f"**🔷 Chunks:** {n_chunks:,}")
    st.markdown(f"**🤖 Model:** `{st.session_state.embedding_model}`")
    st.markdown(f"**✂️ Strategy:** `{st.session_state.chunking_strategy}`")

    st.markdown("---")
    st.markdown("<small style='color:#8b949e'>Built with ❤️ using Streamlit + ChromaDB + HuggingFace</small>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Ask a Question
# ══════════════════════════════════════════════════════════════════════════════

if page == "💬 Ask a Question":
    st.markdown("## 💬 Ask a Question")
    st.markdown("Ask anything about the deep learning papers in the knowledge base. All answers include inline citations.")
    st.markdown("---")

    # Chat history
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(turn["query"])
        with st.chat_message("assistant"):
            st.markdown(
                f'<div class="answer-box">{turn["answer"]}</div>',
                unsafe_allow_html=True,
            )
            if turn.get("citations"):
                with st.expander(f"📎 {len(turn['citations'])} Citation(s)", expanded=False):
                    for i, cit in enumerate(turn["citations"], 1):
                        st.markdown(
                            f"""<div class="citation-block">
                            <span class="cite-title">[{i}] {cit['title']}</span><br>
                            <span class="cite-meta">
                                {', '.join(cit['authors'][:3])} {'et al.' if len(cit['authors']) > 3 else ''} ({cit['year']})
                                {'| ' + cit['section'] if cit['section'] else ''}
                            </span>
                            <div class="cite-excerpt">"{cit['excerpt']} …"</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )
            if turn.get("flagged"):
                with st.expander(f"⚠️ {len(turn['flagged'])} Potentially Unsupported Statement(s)", expanded=False):
                    st.info("These sentences had low entailment scores against the retrieved passages. Verify them manually.")
                    for flag in turn["flagged"]:
                        st.markdown(
                            f"""<div class="flag-block">
                            <div class="flag-sentence">"{flag['sentence']}"</div>
                            <div class="flag-score">Entailment score: {flag['score']:.3f}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )
            if turn.get("not_found"):
                st.warning("⚠️ No sufficiently relevant passages were found for this query.")

    # Input
    query = st.chat_input("Ask a question about deep learning papers …")
    if query:
        retriever = get_retriever(st.session_state.embedding_model, st.session_state.chunking_strategy)
        generator = get_generator()

        if not retriever or retriever.collection_size() == 0:
            st.error("❌ Knowledge base is empty. Go to **📚 Knowledge Base** and ingest papers first.")
            st.stop()

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching knowledge base …"):
                results = retriever.retrieve(
                    query,
                    top_k=cfg["retrieval"]["top_k"],
                    final_k=st.session_state.top_k,
                    threshold=st.session_state.threshold,
                )

            with st.spinner("✍️ Generating answer …"):
                if generator:
                    gen_result = generator.generate(query, results)
                else:
                    from generator import NOT_FOUND_RESPONSE, GenerationResult
                    gen_result = GenerationResult(
                        query=query,
                        answer="Generator not available. Check your API key in .env",
                        citations=[],
                        flagged_sentences=[],
                    )

            st.markdown(
                f'<div class="answer-box">{gen_result.answer}</div>',
                unsafe_allow_html=True,
            )

            cit_dicts = [
                {
                    "title":   c.title,
                    "authors": c.authors,
                    "year":    c.year,
                    "section": c.section,
                    "url":     c.url,
                    "excerpt": c.excerpt,
                }
                for c in gen_result.citations
            ]

            if cit_dicts:
                with st.expander(f"📎 {len(cit_dicts)} Citation(s)", expanded=True):
                    for i, cit in enumerate(cit_dicts, 1):
                        st.markdown(
                            f"""<div class="citation-block">
                            <span class="cite-title">[{i}] {cit['title']}</span><br>
                            <span class="cite-meta">
                                {', '.join(cit['authors'][:3])} {'et al.' if len(cit['authors']) > 3 else ''} ({cit['year']})
                                {'| ' + cit['section'] if cit['section'] else ''}
                            </span>
                            <div class="cite-excerpt">"{cit['excerpt']} …"</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

            if gen_result.flagged_sentences:
                with st.expander(f"⚠️ {len(gen_result.flagged_sentences)} Potentially Unsupported Statement(s)", expanded=True):
                    st.info("These sentences had low entailment scores. Treat with caution.")
                    for flag in gen_result.flagged_sentences:
                        st.markdown(
                            f"""<div class="flag-block">
                            <div class="flag-sentence">"{flag['sentence']}"</div>
                            <div class="flag-score">Entailment score: {flag['score']:.3f}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

            if gen_result.not_found:
                st.warning("⚠️ No sufficiently relevant passages were found for this query.")

            # Context preview
            if results:
                with st.expander(f"🔷 Retrieved {len(results)} Chunk(s)", expanded=False):
                    for r in results:
                        st.markdown(
                            f"""**Rank {r.rank}** | Score: `{r.vector_score:.3f}` | Re-rank: `{r.reranker_score:.3f}`
**{r.chunk.title}** {'[' + r.chunk.section + ']' if r.chunk.section else ''}
""",
                        )
                        st.code(r.chunk.text[:400] + " …" if len(r.chunk.text) > 400 else r.chunk.text, language=None)
                        st.markdown("---")

        # Save to history
        st.session_state.chat_history.append(
            {
                "query":     query,
                "answer":    gen_result.answer,
                "citations": cit_dicts,
                "flagged":   gen_result.flagged_sentences,
                "not_found": gen_result.not_found,
            }
        )

    if st.session_state.chat_history:
        col_l, col_r = st.columns([8, 1])
        with col_r:
            if st.button("🗑️ Clear", help="Clear chat history"):
                st.session_state.chat_history = []
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Knowledge Base
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📚 Knowledge Base":
    st.markdown("## 📚 Knowledge Base")
    st.markdown("---")

    tab_browse, tab_upload, tab_ingest = st.tabs(["📋 Browse Papers", "📤 Upload PDF", "🔄 Re-ingest"])

    with tab_browse:
        papers = load_papers()
        if not papers:
            st.info("No papers found. Upload a PDF or run `python ingest.py` from the terminal.")
        else:
            st.markdown(f"**{len(papers)} papers** in the knowledge base.")
            search = st.text_input("🔎 Filter papers", placeholder="Search by title or author …")
            for p in papers:
                title = p.metadata.title
                authors = ", ".join(p.metadata.authors[:3])
                if len(p.metadata.authors) > 3:
                    authors += " et al."
                year = p.metadata.year
                arxiv_id = p.metadata.arxiv_id
                sections = ", ".join(p.metadata.sections[:4]) if p.metadata.sections else "N/A"

                if search.lower() in title.lower() or search.lower() in authors.lower() or not search:
                    st.markdown(
                        f"""<div class="paper-card">
                        <div class="paper-title">{title}</div>
                        <div class="paper-meta">
                            {authors} · {year} · arXiv:{arxiv_id}<br>
                            <span style="font-size:0.78rem;color:#58a6ff">Sections: {sections}</span>
                        </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

    with tab_upload:
        st.markdown("### Upload a new PDF")
        st.markdown("The uploaded paper will be parsed and added to the knowledge base.")

        uploaded = st.file_uploader("Choose a PDF file", type=["pdf"], key="pdf_uploader")
        if uploaded:
            col_l, col_r = st.columns([2, 1])
            with col_l:
                st.success(f"✅ Selected: **{uploaded.name}** ({uploaded.size // 1024} KB)")
            with col_r:
                if st.button("📥 Ingest Paper", key="ingest_pdf_btn"):
                    with st.spinner("Parsing and indexing …"):
                        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                            tmp.write(uploaded.getvalue())
                            tmp_path = Path(tmp.name)

                        paper = ingest_local_pdf(tmp_path)
                        if paper:
                            processed_dir = ROOT / cfg["paths"]["processed_dir"]
                            save_processed(paper, processed_dir)
                            # Re-index
                            chunks = chunk_all_papers(
                                [paper],
                                strategy=st.session_state.chunking_strategy,
                            )
                            retriever = get_retriever(
                                st.session_state.embedding_model,
                                st.session_state.chunking_strategy,
                            )
                            if retriever:
                                retriever.index(chunks)
                            # Clear caches so sidebar stats update
                            st.cache_resource.clear()
                            st.success(f"✅ **{paper.metadata.title}** ingested and indexed!")
                        else:
                            st.error("❌ Failed to parse PDF. Try a different file.")
                        tmp_path.unlink(missing_ok=True)

    with tab_ingest:
        st.markdown("### Re-ingest from arXiv")
        st.markdown("Run the ingestion pipeline to download/re-process papers from arXiv.")
        st.code("python ingest.py --n 35", language="bash")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Re-index existing papers**")
            st.markdown("Use this if you've changed the embedding model or chunking strategy.")
            if st.button("🔄 Re-index Knowledge Base"):
                with st.spinner("Re-indexing …"):
                    st.cache_resource.clear()
                    papers = load_papers()
                    chunks = chunk_all_papers(papers, strategy=st.session_state.chunking_strategy)
                    retriever = Retriever(
                        embedding_model_slug=st.session_state.embedding_model,
                        chunking_strategy=st.session_state.chunking_strategy,
                    )
                    retriever.delete_collection()
                    retriever.index(chunks)
                st.success(f"✅ Re-indexed {len(chunks):,} chunks.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: Settings
# ══════════════════════════════════════════════════════════════════════════════

elif page == "⚙️ Settings":
    st.markdown("## ⚙️ Settings")
    st.markdown("Changes here take effect immediately for new queries. Re-indexing is needed when switching models or chunking strategies.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🤖 Retrieval")
        model_options = list(cfg["embeddings"]["models"].keys())
        new_model = st.selectbox(
            "Embedding Model",
            model_options,
            index=model_options.index(st.session_state.embedding_model),
            key="settings_model",
        )

        strategy_options = ["fixed", "recursive", "section"]
        new_strategy = st.selectbox(
            "Chunking Strategy",
            strategy_options,
            index=strategy_options.index(st.session_state.chunking_strategy),
            key="settings_strategy",
        )

        new_top_k = st.slider("Top-k chunks sent to LLM", 1, 10, st.session_state.top_k, key="settings_k")
        new_threshold = st.slider(
            "Similarity threshold",
            0.0, 1.0,
            st.session_state.threshold,
            step=0.05,
            key="settings_threshold",
        )
        new_reranker = st.toggle("Use cross-encoder re-ranker", value=st.session_state.use_reranker)

    with col2:
        st.markdown("### 🔤 Model Details")
        model_info = cfg["embeddings"]["models"][st.session_state.embedding_model]
        st.markdown(f"""
| Property | Value |
|---|---|
| HuggingFace name | `{model_info['hf_name']}` |
| Embedding dimension | `{model_info['dimension']}` |
| Batch size | `{model_info['batch_size']}` |
| Query prefix | `{model_info.get('query_prefix', 'None')}` |
""")
        st.markdown("### 🔧 LLM Provider")
        active_provider = cfg["generation"].get("llm_provider", "extractive")
        active_model = cfg["generation"].get(active_provider, {}).get("model", "Extractive / Context-grounded")
        active_temp = cfg["generation"].get(active_provider, {}).get("temperature", "N/A")
        st.markdown(f"""
| Property | Value |
|---|---|
| Provider | `{active_provider}` |
| Model | `{active_model}` |
| Temperature | `{active_temp}` |
""")
        st.info("To change the LLM provider, edit `config.yaml` or set your API key in `.env`.")

    st.markdown("---")
    if st.button("💾 Apply Settings"):
        changed_model    = new_model    != st.session_state.embedding_model
        changed_strategy = new_strategy != st.session_state.chunking_strategy

        st.session_state.embedding_model   = new_model
        st.session_state.chunking_strategy = new_strategy
        st.session_state.top_k             = new_top_k
        st.session_state.threshold         = new_threshold
        st.session_state.use_reranker      = new_reranker

        if changed_model or changed_strategy:
            st.warning("⚠️ Embedding model or chunking strategy changed — re-indexing may be required. Go to **📚 Knowledge Base → Re-ingest**.")
        else:
            st.success("✅ Settings applied.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: Evaluation Dashboard
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Evaluation":
    st.markdown("## 📊 Evaluation Dashboard")
    st.markdown("---")

    eval_dir = ROOT / cfg["paths"]["eval_dir"]

    tab_metrics, tab_grid, tab_failures = st.tabs(
        ["📈 Retrieval Metrics", "🔲 Comparison Grid", "🔍 Failure Analysis"]
    )

    with tab_metrics:
        metrics_file = eval_dir / "retrieval_metrics.json"
        faith_file   = eval_dir / "faithfulness_metrics.json"

        if not metrics_file.exists():
            st.info("No evaluation results yet. Run `python evaluate.py` from the terminal.")
            st.code("python evaluate.py", language="bash")
        else:
            with open(metrics_file) as f:
                metrics = json.load(f)

            cols = st.columns(4)
            metric_items = [
                ("recall@1",  "Recall@1",  "#58a6ff"),
                ("recall@3",  "Recall@3",  "#3fb950"),
                ("recall@5",  "Recall@5",  "#bc8cff"),
                ("mrr",       "MRR",       "#f78166"),
            ]
            for col, (key, label, color) in zip(cols, metric_items):
                with col:
                    val = metrics.get(key, 0.0)
                    st.markdown(
                        f"""<div class="metric-card">
                        <div class="label">{label}</div>
                        <div class="value" style="color:{color}">{val:.3f}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

            if faith_file.exists():
                with open(faith_file) as f:
                    faith = json.load(f)
                st.markdown("---")
                st.markdown("### Answer Faithfulness")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(
                        f"""<div class="metric-card">
                        <div class="label">Faithfulness</div>
                        <div class="value" style="color:#3fb950">{faith.get('faithfulness', 0):.3f}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.markdown(
                        f"""<div class="metric-card">
                        <div class="label">Avg Flagged / Answer</div>
                        <div class="value" style="color:#f78166">{faith.get('avg_flagged', 0):.2f}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                with c3:
                    st.markdown(
                        f"""<div class="metric-card">
                        <div class="label">Answers Evaluated</div>
                        <div class="value">{faith.get('n_evaluated', 0)}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

    with tab_grid:
        grid_file = eval_dir / "grid_results.csv"
        grid_img  = next(eval_dir.glob("grid_recall_at_*.png"), None)

        if not grid_file.exists():
            st.info("Grid results not available. Run `python evaluate.py --grid` from the terminal.")
            st.code("python evaluate.py --grid", language="bash")
        else:
            import pandas as pd
            df = pd.read_csv(grid_file)
            st.markdown("### Recall@k Comparison Grid")
            st.dataframe(df.style.background_gradient(cmap="YlGnBu", subset=[c for c in df.columns if "recall" in c or "mrr" in c]))

            if grid_img:
                st.image(str(grid_img), caption="Recall@k Heatmap", use_column_width=True)

    with tab_failures:
        failures_file = eval_dir / "retrieval_failures.json"
        if not failures_file.exists():
            st.info("No failure log yet. Run the evaluation first.")
        else:
            with open(failures_file) as f:
                failures = json.load(f)
            st.markdown(f"### {len(failures)} Retrieval Failure(s)")
            for i, fail in enumerate(failures, 1):
                with st.expander(f"❌ [{i}] {fail['question'][:80]}…"):
                    st.markdown(f"**Expected paper:** `{fail['expected']}`")
                    st.markdown(f"**Top retrieved:** {fail.get('top_retrieved', [])}")
                    scores = fail.get("top_scores", [])
                    if scores:
                        st.markdown(f"**Top scores:** {[f'{s:.3f}' for s in scores]}")
                    st.markdown(f"**Diagnosis:** {fail.get('diagnosis', 'N/A')}")

    st.markdown("---")
    st.markdown("### ▶️ Run Evaluation")
    col_a, col_b = st.columns(2)
    with col_a:
        st.code("python evaluate.py", language="bash")
        st.caption("Basic retrieval evaluation")
    with col_b:
        st.code("python evaluate.py --grid --no-gen", language="bash")
        st.caption("Full grid (no LLM calls)")
