#!/usr/bin/env python3
"""
Streamlit Web Application for arXiv MCP Research Server.
Provides a modern web UI for searching, analysing, and exporting
academic research papers from arXiv.
"""

import asyncio
import html
import json
import os
import re
import subprocess
import time
from datetime import datetime
from typing import Any, Dict, List

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (
    logger,
    PAPER_DIR,
    search_arxiv,
    save_papers,
    load_papers,
    list_saved_topics,
    topic_dir_name,
    extract_keywords,
    extract_key_contributions,
    extract_methodologies,
    categorize_papers,
    papers_by_year,
    papers_to_bibtex,
    papers_to_markdown_table,
    papers_to_apa_citations,
    build_literature_review,
    build_trend_report,
    build_author_network,
    # RAG / AI features
    expand_query,
    semantic_search_all,
    build_rag_context,
    compute_paper_similarity,
    analyze_research_gaps,
    generate_research_questions,
    run_research_agent,
    build_corpus_chunks,
    TFIDFIndex,
    # Advanced AI features
    hybrid_search,
    textrank_summarize,
    extract_entities,
    cluster_papers,
    analyze_topic_drift,
    format_topic_drift,
    build_knowledge_graph,
    format_knowledge_graph,
)

# =========================================================================
# Page config
# =========================================================================
st.set_page_config(
    page_title="arXiv Research Assistant",
    page_icon="\U0001f52c",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================================
# CSS
# =========================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem; font-weight: 700;
        text-align: center; margin-bottom: 1.5rem; padding: 1.2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,.15);
    }
    .card {
        background: white; padding: 1.2rem; border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,.08); margin: .8rem 0;
        border-left: 4px solid #667eea;
    }
    .metric-card {
        background: white; padding: 1rem; border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,.08); text-align: center;
    }
    .chat-msg { padding: 1rem; border-radius: .5rem; margin-bottom: 1rem; }
    .user-msg  { background: #f0f2f6; border-left: 4px solid #ff6b6b; }
    .asst-msg  { background: #e8f4fd; border-left: 4px solid #667eea; }
</style>
""", unsafe_allow_html=True)

# =========================================================================
# Session state defaults
# =========================================================================
_defaults = {
    "chat_history": [],
    "server_running": False,
    "server_proc": None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =========================================================================
# Helper: run async from sync Streamlit
# =========================================================================
def _run_async(coro):
    """Execute an async coroutine from synchronous Streamlit code."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


# =========================================================================
# Visualisation builders (Plotly)
# =========================================================================

def _fig_timeline(papers: Dict[str, Dict[str, Any]]) -> "go.Figure | None":
    """Bar chart of papers per publication year."""
    by_yr = papers_by_year(papers)
    years = sorted(by_yr.keys())
    counts = [len(by_yr[y]) for y in years]
    fig = px.bar(x=years, y=counts, labels={"x": "Year", "y": "Papers"},
                 title="Publication Timeline", color=counts,
                 color_continuous_scale="Viridis")
    fig.update_layout(showlegend=False, height=350)
    return fig


def _fig_categories(papers: Dict[str, Dict[str, Any]]) -> "go.Figure | None":
    """Pie chart of arXiv category distribution."""
    from collections import Counter
    cat_counts = Counter()
    for p in papers.values():
        for c in p.get("categories", []):
            cat_counts[c] += 1
    if not cat_counts:
        return None
    labels, values = zip(*cat_counts.most_common(10))
    fig = px.pie(names=labels, values=values, title="Category Distribution",
                 hole=0.35)
    fig.update_layout(height=350)
    return fig


def _fig_keywords(papers: Dict[str, Dict[str, Any]]) -> "go.Figure | None":
    """Horizontal bar chart of top keywords."""
    summaries = [p["summary"] for p in papers.values()]
    kws = extract_keywords(summaries, top_n=12)
    if not kws:
        return None
    words, counts = zip(*reversed(kws))
    fig = px.bar(x=list(counts), y=list(words), orientation="h",
                 labels={"x": "Frequency", "y": "Keyword"},
                 title="Top Keywords", color=list(counts),
                 color_continuous_scale="Blues")
    fig.update_layout(showlegend=False, height=400)
    return fig


def _fig_methods(papers: Dict[str, Dict[str, Any]]) -> "go.Figure | None":
    """Bar chart of methodology distribution."""
    methods = extract_methodologies(papers)
    if not methods:
        return None
    labels, values = zip(*methods.items())
    fig = px.bar(x=list(labels), y=list(values),
                 labels={"x": "Methodology", "y": "Papers"},
                 title="Methodology Distribution", color=list(values),
                 color_continuous_scale="Sunset")
    fig.update_layout(showlegend=False, height=350)
    return fig


def _fig_author_collab(papers: Dict[str, Dict[str, Any]]) -> "go.Figure | None":
    """Bar chart of most-active authors."""
    from collections import Counter
    ac = Counter()
    for p in papers.values():
        for a in p["authors"]:
            ac[a] += 1
    top = ac.most_common(10)
    if not top:
        return None
    names, counts = zip(*reversed(top))
    fig = px.bar(x=list(counts), y=list(names), orientation="h",
                 labels={"x": "Papers", "y": "Author"},
                 title="Most Active Authors", color=list(counts),
                 color_continuous_scale="Teal")
    fig.update_layout(showlegend=False, height=400)
    return fig


# =========================================================================
# Chat message renderer
# =========================================================================

def _append_and_rerun(role: str, content: str):
    st.session_state.chat_history.append({"type": role, "content": content})
    st.rerun()

def _render_chat():
    for msg in st.session_state.chat_history:
        if msg["type"] == "user":
            safe = html.escape(msg["content"])
            st.markdown(f'<div class="chat-msg user-msg"><strong>You:</strong><br>{safe}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-msg asst-msg"><strong>Assistant:</strong></div>',
                        unsafe_allow_html=True)
            st.markdown(msg["content"])


# =========================================================================
# Route user input to the right action
# =========================================================================

def _route_input(text: str):
    """Parse free-text user input and dispatch to the right tool."""
    low = text.lower().strip()

    # --- @commands ---
    if low == "@folders":
        topics = list_saved_topics()
        if topics:
            res = "# Available Topics\n\n" + "\n".join(f"- {t}" for t in topics)
        else:
            res = "No topics found. Search for papers first!"
        return res

    if low.startswith("@"):
        topic = text[1:].strip()
        papers = load_papers(topic)
        if papers:
            return papers_to_markdown_table(papers)
        return f"No papers found for '{topic}'. Search first."

    # --- Extract topic with regex ---
    topic = None
    for pattern in [
        r"(?:search|find)\s+(?:for\s+)?(?:papers?|research)\s+(?:on|about|for)\s+(.+)",
        r"(?:literature\s+review|analyze|analyse|summarize|summarise)\s+(?:papers?\s+)?(?:on|about|for)\s+(.+)",
        r"(?:research\s+proposal|proposal)\s+(?:for|on|about)\s+(.+)",
        r"(?:trends?|trend\s+analysis)\s+(?:for|on|about|in)\s+(.+)",
        r"(?:author\s+network|authors?)\s+(?:for|on|about)\s+(.+)",
        r"(?:citation|citations?)\s+(?:for|on|about)\s+(.+)",
        r"(?:export|bibtex|bib)\s+(?:for|on|about)\s+(.+)",
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            topic = m.group(1).strip().strip("'\"")
            break

    if not topic or len(topic) < 2:
        return _help_text()

    # --- Dispatch ---
    if any(k in low for k in ["search", "find"]):
        papers = _run_async(search_arxiv(topic, 5))
        if not papers:
            return f"No papers found for '{topic}'. Try broader terms."
        save_papers(topic, papers)
        lines = [f"Found **{len(papers)}** papers on *{topic}*:\n"]
        for i, (pid, p) in enumerate(papers.items(), 1):
            lines.append(f"### {i}. {p['title']}")
            lines.append(f"**Authors:** {', '.join(p['authors'])}")
            lines.append(f"**Published:** {p['published']}")
            ab = p["summary"][:200] + ("..." if len(p["summary"]) > 200 else "")
            lines.append(f"**Abstract:** {ab}")
            lines.append(f"[PDF]({p['pdf_url']})\n")
        return "\n".join(lines)

    if any(k in low for k in ["literature review", "analyze", "analyse", "summarize", "summarise"]):
        papers = load_papers(topic)
        if not papers:
            return f"No papers for '{topic}'. Search first."
        return build_literature_review(topic, papers)

    if "proposal" in low:
        papers = load_papers(topic)
        if not papers:
            return f"No papers for '{topic}'. Search first."
        return build_literature_review(topic, papers) + "\n\n---\n*Use this review as the foundation for your research proposal.*"

    if any(k in low for k in ["trend", "timeline"]):
        papers = load_papers(topic)
        if not papers:
            return f"No papers for '{topic}'. Search first."
        return build_trend_report(topic, papers)

    if any(k in low for k in ["author", "network", "collaborat"]):
        papers = load_papers(topic)
        if not papers:
            return f"No papers for '{topic}'. Search first."
        return build_author_network(topic, papers)

    if any(k in low for k in ["export", "bibtex", "citation"]):
        papers = load_papers(topic)
        if not papers:
            return f"No papers for '{topic}'. Search first."
        return "## BibTeX\n```bibtex\n" + papers_to_bibtex(papers) + "\n```\n\n## APA\n" + papers_to_apa_citations(papers)

    return _help_text()


def _help_text() -> str:
    return """**How to use this assistant:**

| Command | Example |
|---------|---------|
| **Search** | "Search for papers on machine learning" |
| **Literature Review** | "Analyze papers on deep learning" |
| **Trends** | "Show trends for quantum computing" |
| **Authors** | "Author network for NLP" |
| **Export** | "Export bibtex for reinforcement learning" |
| **Research Proposal** | "Research proposal for federated learning" |
"""


# =========================================================================
# MAIN
# =========================================================================

def main():
    # --- Header ---
    st.markdown('<div class="main-header">\U0001f52c arXiv Research Assistant</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align:center;color:#666;font-size:1.1rem;">'
        '<b>AI-Powered Academic Research Platform</b> &mdash; '
        'Search, analyse, visualise &amp; export research papers from arXiv</p>',
        unsafe_allow_html=True)

    # --- Status row ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data Source", "arXiv API")
    c2.metric("Cached Topics", str(len(list_saved_topics())))
    total_papers = sum(len(load_papers(t)) for t in list_saved_topics())
    c3.metric("Total Papers", str(total_papers))
    c4.metric("Export Formats", "BibTeX / APA / MD")

    st.markdown("---")

    # === TAB LAYOUT ========================================================
    tab_search, tab_rag, tab_viz, tab_export, tab_chat = st.tabs(
        ["\U0001f50d Search & Analyse", "\U0001f9e0 AI / RAG",
         "\U0001f4ca Visualisations", "\U0001f4e4 Export", "\U0001f4ac Chat"])

    # ----- TAB: Search & Analyse ------------------------------------------
    with tab_search:
        col_in, col_n = st.columns([4, 1])
        with col_in:
            search_topic = st.text_input(
                "Research Topic",
                placeholder="e.g. transformer architectures, federated learning ...",
                key="search_input")
        with col_n:
            num_papers = st.number_input("Papers", 1, 20, 5, key="num_papers")

        b1, b2, b3, b4 = st.columns(4)
        do_search   = b1.button("\U0001f50d Search Papers", type="primary", width="stretch")
        do_review   = b2.button("\U0001f4dd Literature Review", width="stretch")
        do_trends   = b3.button("\U0001f4c8 Trend Analysis", width="stretch")
        do_authors  = b4.button("\U0001f465 Author Network", width="stretch")

        if do_search and search_topic:
            with st.spinner(f"Searching arXiv for '{search_topic}' ..."):
                papers = _run_async(search_arxiv(search_topic, num_papers))
            if papers:
                save_papers(search_topic, papers)
                st.success(f"Found {len(papers)} papers!")
                for i, (pid, p) in enumerate(papers.items(), 1):
                    with st.expander(f"{i}. {p['title']}", expanded=(i <= 2)):
                        st.markdown(f"**Authors:** {', '.join(p['authors'])}")
                        st.markdown(f"**Published:** {p['published']}  |  **Categories:** {', '.join(p.get('categories', []))}")
                        st.markdown(f"**Abstract:** {p['summary']}")
                        contribs = extract_key_contributions(p["summary"])
                        if contribs:
                            st.markdown("**Key Contributions:**")
                            for c in contribs:
                                st.markdown(f"- {c}")
                        st.markdown(f"[\U0001f4c4 PDF]({p['pdf_url']})  |  [\U0001f517 Abstract]({p.get('abs_url', '')})")
            else:
                st.warning("No results. Try different keywords.")

        if do_review and search_topic:
            papers = load_papers(search_topic)
            if papers:
                st.markdown(build_literature_review(search_topic, papers))
            else:
                st.warning("No saved papers. Search first!")

        if do_trends and search_topic:
            papers = load_papers(search_topic)
            if papers:
                st.markdown(build_trend_report(search_topic, papers))
            else:
                st.warning("No saved papers. Search first!")

        if do_authors and search_topic:
            papers = load_papers(search_topic)
            if papers:
                st.markdown(build_author_network(search_topic, papers))
            else:
                st.warning("No saved papers. Search first!")

    # ----- TAB: AI / RAG ---------------------------------------------------
    with tab_rag:
        st.markdown("### \U0001f9e0 AI-Powered Research Intelligence")
        st.caption(
            "TF-IDF retrieval, gap analysis, similarity mapping, "
            "research question generation, and autonomous agent workflow."
        )

        rag_topics = list_saved_topics()
        if not rag_topics:
            st.info("Search & save papers first — the RAG engine indexes your saved corpus.")
        else:
            # --- Semantic search across all saved papers ---
            st.markdown("---")
            st.markdown("#### \U0001f50e Cross-Topic Semantic Search")
            rag_query = st.text_input(
                "Search across all saved papers",
                placeholder="e.g. attention mechanism for time-series",
                key="rag_query",
            )
            rag_k = st.slider("Results", 3, 15, 6, key="rag_k")
            if st.button("\U0001f50d Semantic Search", key="btn_sem_search") and rag_query:
                with st.spinner("Building TF-IDF index across all topics …"):
                    results = semantic_search_all(rag_query, top_k=rag_k)
                if results:
                    for r in results:
                        score_pct = r["score"] * 100
                        color = "#27ae60" if score_pct > 30 else "#f39c12" if score_pct > 10 else "#e74c3c"
                        st.markdown(
                            f'<div class="card">'
                            f'<strong>{r.get("title", "Untitled")}</strong> '
                            f'<span style="color:{color};font-weight:600">'
                            f'({score_pct:.1f}% relevance)</span><br>'
                            f'<em>Paper: {r.get("paper_id", "")}</em><br>'
                            f'{r.get("text", "")}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.warning("No matching passages. Try different keywords.")

            # --- RAG Context Builder ---
            st.markdown("---")
            st.markdown("#### \U0001f4da RAG Context Builder")
            st.caption(
                "Retrieves the most relevant chunks from your paper corpus "
                "— ready to paste into an LLM prompt."
            )
            rag_ctx_query = st.text_input(
                "Question / prompt",
                placeholder="e.g. What are the main challenges in federated NLP?",
                key="rag_ctx_query",
            )
            if st.button("\U0001f4cb Build Context", key="btn_rag_ctx") and rag_ctx_query:
                with st.spinner("Retrieving & assembling context …"):
                    ctx = build_rag_context(rag_ctx_query, top_k=6)
                st.markdown(ctx)
                st.download_button(
                    "Download context (.md)",
                    data=ctx,
                    file_name="rag_context.md",
                    mime="text/markdown",
                )

            # --- Topic-scoped analysis ---
            st.markdown("---")
            st.markdown("#### Analysis for a Specific Topic")
            rag_topic = st.selectbox(
                "Select topic", rag_topics, key="rag_topic"
            )
            papers = load_papers(rag_topic)

            if papers:
                col_gap, col_q = st.columns(2)

                # Gap analysis
                with col_gap:
                    st.markdown("##### \U0001f6a6 Research Gap Analysis")
                    if st.button("Run Gap Analysis", key="btn_gaps"):
                        with st.spinner("Scanning abstracts against 9 dimensions …"):
                            gaps = analyze_research_gaps(papers)
                        _STATUS_ICON = {"gap": "\U0001f534", "under-explored": "\U0001f7e1", "covered": "\U0001f7e2"}
                        for g in gaps:
                            icon = _STATUS_ICON.get(g["status"], "\u26aa")
                            st.markdown(f"{icon} **{g['dimension']}** — *{g['status']}*")
                            st.caption(g["detail"])

                # Research questions
                with col_q:
                    st.markdown("##### \u2753 Generated Research Questions")
                    if st.button("Generate Questions", key="btn_questions"):
                        with st.spinner("Analysing gaps + keywords …"):
                            questions = generate_research_questions(rag_topic, papers)
                        if questions:
                            for i, q in enumerate(questions, 1):
                                st.markdown(f"**{i}.** {q}")
                        else:
                            st.info("Not enough data to generate questions.")

                # Similarity heatmap
                st.markdown("---")
                st.markdown("##### \U0001f5fa Paper Similarity Heatmap")
                if st.button("Compute Similarity", key="btn_sim"):
                    with st.spinner("Building pairwise TF-IDF similarity matrix …"):
                        sim = compute_paper_similarity(papers)
                    if sim:
                        pids = list(sim.keys())
                        labels = [papers[pid]["title"][:50] + "…" if len(papers[pid]["title"]) > 50 else papers[pid]["title"] for pid in pids]
                        # Build NxN matrix
                        n = len(pids)
                        matrix = [[0.0] * n for _ in range(n)]
                        for i, pid_i in enumerate(pids):
                            for entry in sim[pid_i]:
                                j_pid = entry["paper_id"]
                                if j_pid in pids:
                                    j = pids.index(j_pid)
                                    matrix[i][j] = entry["score"]
                            matrix[i][i] = 1.0  # self-similarity
                        fig = go.Figure(
                            data=go.Heatmap(
                                z=matrix,
                                x=labels,
                                y=labels,
                                colorscale="Viridis",
                                hoverongaps=False,
                            )
                        )
                        fig.update_layout(
                            title="Pairwise Abstract Similarity (TF-IDF Cosine)",
                            height=500,
                            xaxis_tickangle=-45,
                        )
                        st.plotly_chart(fig, width="stretch")

                        # Top similar pairs
                        st.markdown("**Top similar pairs:**")
                        pairs_seen: set = set()
                        ranked_pairs = []
                        for pid_i in pids:
                            for entry in sim[pid_i]:
                                pair_key = tuple(sorted([pid_i, entry["paper_id"]]))
                                if pair_key not in pairs_seen:
                                    pairs_seen.add(pair_key)
                                    ranked_pairs.append((pid_i, entry))
                        ranked_pairs.sort(key=lambda x: x[1]["score"], reverse=True)
                        for pid_i, entry in ranked_pairs[:5]:
                            st.markdown(
                                f"- **{papers[pid_i]['title'][:60]}** ↔ "
                                f"**{entry['title'][:60]}** — "
                                f"similarity {entry['score']:.3f}"
                            )
                    else:
                        st.info("Need at least 2 papers for similarity analysis.")

                # --- Hybrid Search (TF-IDF + BM25 + RRF) ---
                st.markdown("---")
                st.markdown("##### \U0001f504 Hybrid Search (TF-IDF + BM25 + RRF)")
                st.caption(
                    "Two-stage retrieval: TF-IDF recall + BM25 recall → Reciprocal Rank Fusion. "
                    "Production-grade pattern used by search engines."
                )
                hybrid_q = st.text_input(
                    "Hybrid search query",
                    placeholder="e.g. privacy-preserving transformer",
                    key="hybrid_q",
                )
                if st.button("\U0001f50d Hybrid Search", key="btn_hybrid") and hybrid_q:
                    with st.spinner("TF-IDF + BM25 → Reciprocal Rank Fusion …"):
                        h_results = hybrid_search(hybrid_q, top_k=8)
                    if h_results:
                        for r in h_results:
                            rrf = r.get("rrf_score", 0)
                            st.markdown(
                                f'<div class="card">'
                                f'<strong>{r.get("title", "")}</strong> '
                                f'<span style="color:#2980b9;font-weight:600">'
                                f'(RRF: {rrf:.4f})</span><br>'
                                f'<em>{r.get("paper_id", "")}</em><br>'
                                f'{r.get("text", "")}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.warning("No results.")

                # --- TextRank Summariser ---
                st.markdown("---")
                col_tr, col_ner = st.columns(2)
                with col_tr:
                    st.markdown("##### \U0001f4dd TextRank Summary")
                    st.caption("Graph-based multi-document extractive summarisation (PageRank on sentences).")
                    tr_n = st.slider("Sentences", 4, 12, 8, key="tr_n")
                    if st.button("Summarise", key="btn_textrank"):
                        with st.spinner("Running TextRank (PageRank on sentence graph) …"):
                            summary = textrank_summarize(papers, num_sentences=tr_n)
                        st.markdown(summary)

                # --- Named Entity Extraction ---
                with col_ner:
                    st.markdown("##### \U0001f3f7 Named Entity Extraction")
                    st.caption("Regex-based NER: AI models, datasets, metrics, and tasks.")
                    if st.button("Extract Entities", key="btn_ner"):
                        with st.spinner("Scanning abstracts for entities …"):
                            entities = extract_entities(papers)
                        for cat in ["models", "datasets", "metrics", "tasks"]:
                            items = entities.get(cat, [])
                            st.markdown(f"**{cat.title()}** ({len(items)})")
                            if items:
                                for e in items[:8]:
                                    st.markdown(
                                        f"- `{e['name']}` — {e['count']} mention(s), "
                                        f"{len(e['papers'])} paper(s)"
                                    )
                            else:
                                st.caption("None detected.")

                # --- Paper Clustering ---
                st.markdown("---")
                col_cl, col_drift = st.columns(2)
                with col_cl:
                    st.markdown("##### \U0001f4e6 Paper Clustering (K-Means)")
                    st.caption("Unsupervised grouping on TF-IDF vectors with auto-labeled clusters.")
                    n_cl = st.slider("Clusters", 2, min(8, max(2, len(papers))), min(3, len(papers)), key="n_cl")
                    if st.button("Cluster Papers", key="btn_cluster"):
                        with st.spinner("Running K-Means on TF-IDF vectors …"):
                            clusters = cluster_papers(papers, n_clusters=n_cl)
                        for c in clusters:
                            with st.expander(f"Cluster {c['cluster_id']}: {c['label']}", expanded=True):
                                st.markdown(f"**{c['size']} paper(s)** | Keywords: {', '.join(c['keywords'])}")
                                for p in c["papers"]:
                                    st.markdown(f"- {p['title']}")

                # --- Topic Drift ---
                with col_drift:
                    st.markdown("##### \U0001f4c8 Topic Evolution / Drift")
                    st.caption("Track how research keywords shift over time.")
                    if st.button("Analyse Drift", key="btn_drift"):
                        with st.spinner("Analysing temporal keyword shifts …"):
                            drift_md = format_topic_drift(papers)
                        st.markdown(drift_md)

                # --- Knowledge Graph ---
                st.markdown("---")
                st.markdown("##### \U0001f578 Knowledge Graph")
                st.caption(
                    "Concept co-occurrence network — which ideas appear together across papers."
                )
                if st.button("Build Knowledge Graph", key="btn_kg"):
                    with st.spinner("Extracting concepts & building co-occurrence graph …"):
                        kg = build_knowledge_graph(papers)
                        kg_md = format_knowledge_graph(papers)
                    st.markdown(kg_md)

                    # Visualise as a network using Plotly
                    if kg["edges"]:
                        import math as _math
                        nodes = kg["nodes"]
                        node_ids = [n["id"] for n in nodes]
                        # Simple circular layout
                        n_nodes = len(node_ids)
                        angle_step = 2 * _math.pi / max(n_nodes, 1)
                        pos = {
                            nid: (_math.cos(i * angle_step), _math.sin(i * angle_step))
                            for i, nid in enumerate(node_ids)
                        }
                        edge_x, edge_y = [], []
                        for e in kg["edges"]:
                            x0, y0 = pos.get(e["source"], (0, 0))
                            x1, y1 = pos.get(e["target"], (0, 0))
                            edge_x.extend([x0, x1, None])
                            edge_y.extend([y0, y1, None])
                        node_x = [pos[nid][0] for nid in node_ids]
                        node_y = [pos[nid][1] for nid in node_ids]
                        node_sizes = [n["weight"] * 8 + 10 for n in nodes]
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=edge_x, y=edge_y, mode="lines",
                            line=dict(width=0.8, color="#999"),
                            hoverinfo="none",
                        ))
                        fig.add_trace(go.Scatter(
                            x=node_x, y=node_y, mode="markers+text",
                            text=node_ids, textposition="top center",
                            marker=dict(size=node_sizes, color=node_sizes,
                                        colorscale="Viridis", showscale=True,
                                        colorbar=dict(title="Mentions")),
                            hovertext=[f"{n['id']}: {n['weight']} mentions" for n in nodes],
                        ))
                        fig.update_layout(
                            title="Concept Co-Occurrence Network",
                            showlegend=False, height=550,
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        )
                        st.plotly_chart(fig, width="stretch")

                # Autonomous Agent
                st.markdown("---")
                st.markdown("##### \U0001f916 Autonomous Research Agent")
                st.caption(
                    "Runs a full 8-step pipeline: query expansion → arXiv search → "
                    "chunking → indexing → retrieval → gap analysis → question "
                    "generation → literature review."
                )
                agent_topic = st.text_input(
                    "Agent topic (defaults to selected topic above)",
                    value=rag_topic,
                    key="agent_topic",
                )
                agent_n = st.slider("Max papers", 3, 15, 8, key="agent_n")
                if st.button("\U0001f680 Run Agent", type="primary", key="btn_agent"):
                    with st.spinner("Running autonomous research pipeline — this may take a moment …"):
                        report = _run_async(run_research_agent(agent_topic, max_papers=agent_n))
                    st.markdown(report)
                    st.download_button(
                        "Download Agent Report (.md)",
                        data=report,
                        file_name=f"agent_report_{agent_topic.replace(' ', '_')}.md",
                        mime="text/markdown",
                    )

    # ----- TAB: Visualisations --------------------------------------------
    with tab_viz:
        viz_topics = list_saved_topics()
        if not viz_topics:
            st.info("Search for papers first to see visualisations.")
        else:
            viz_topic = st.selectbox("Select topic", viz_topics, key="viz_topic")
            papers = load_papers(viz_topic)
            if papers:
                col_a, col_b = st.columns(2)
                with col_a:
                    fig = _fig_timeline(papers)
                    if fig: st.plotly_chart(fig, width="stretch")
                    fig = _fig_keywords(papers)
                    if fig: st.plotly_chart(fig, width="stretch")
                with col_b:
                    fig = _fig_categories(papers)
                    if fig: st.plotly_chart(fig, width="stretch")
                    fig = _fig_methods(papers)
                    if fig: st.plotly_chart(fig, width="stretch")

                st.markdown("### Author Activity")
                fig = _fig_author_collab(papers)
                if fig: st.plotly_chart(fig, width="stretch")

    # ----- TAB: Export -----------------------------------------------------
    with tab_export:
        exp_topics = list_saved_topics()
        if not exp_topics:
            st.info("Search for papers first to enable exports.")
        else:
            exp_topic = st.selectbox("Select topic", exp_topics, key="exp_topic")
            papers = load_papers(exp_topic)
            if papers:
                e1, e2, e3 = st.columns(3)
                with e1:
                    bib = papers_to_bibtex(papers)
                    st.download_button("\u2b07 Download BibTeX",
                                       data=bib, file_name=f"{exp_topic}.bib",
                                       mime="application/x-bibtex",
                                       width="stretch")
                with e2:
                    apa = papers_to_apa_citations(papers)
                    st.download_button("\u2b07 Download APA Citations",
                                       data=apa, file_name=f"{exp_topic}_apa.txt",
                                       mime="text/plain",
                                       width="stretch")
                with e3:
                    review_md = build_literature_review(exp_topic, papers)
                    st.download_button("\u2b07 Download Full Review (MD)",
                                       data=review_md,
                                       file_name=f"{exp_topic}_review.md",
                                       mime="text/markdown",
                                       width="stretch")

                st.markdown("### Preview: BibTeX")
                st.code(bib[:1500] + ("\n..." if len(bib) > 1500 else ""), language="bibtex")

                st.markdown("### Preview: APA")
                st.markdown(apa)

                st.markdown("### Preview: Markdown Table")
                st.markdown(papers_to_markdown_table(papers))

    # ----- TAB: Chat -------------------------------------------------------
    with tab_chat:
        _render_chat()

        user_input = st.text_area(
            "Type your message ...",
            placeholder="e.g. 'Search for papers on graph neural networks'",
            key="chat_input")

        bc1, bc2 = st.columns([3, 1])
        send = bc1.button("\U0001f4e4 Send", type="primary", width="stretch")
        clear = bc2.button("\U0001f9f9 Clear Chat", width="stretch")

        if clear:
            st.session_state.chat_history = []
            st.rerun()

        if send and user_input:
            st.session_state.chat_history.append({"type": "user", "content": user_input})
            with st.spinner("Processing ..."):
                response = _route_input(user_input)
            st.session_state.chat_history.append({"type": "assistant", "content": response})
            st.rerun()


if __name__ == "__main__":
    main()
