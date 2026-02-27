#!/usr/bin/env python3
"""
MCP Research Server for arXiv Paper Search and Analysis.

Exposes tools, resources, and prompts via FastMCP so that any
MCP-compatible client (Claude Desktop, Cursor, etc.) can search,
analyse, and synthesise academic literature from arXiv.
"""

import os
from typing import Dict, List, Any
from datetime import datetime

from fastmcp import FastMCP

# Shared helpers (caching, API, NLP, export, RAG)
from utils import (
    logger,
    PAPER_DIR,
    search_arxiv,
    save_papers,
    load_papers,
    list_saved_topics,
    extract_keywords,
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
    find_similar_papers,
    analyze_research_gaps,
    generate_research_questions,
    run_research_agent,
    build_corpus_chunks,
    # Advanced AI features
    hybrid_search,
    textrank_summarize,
    extract_entities,
    cluster_papers,
    format_topic_drift,
    format_knowledge_graph,
)

# -- MCP Server -----------------------------------------------------------
mcp = FastMCP("arXiv Research Server")


# =========================================================================
# TOOLS
# =========================================================================

@mcp.tool()
async def search_papers(topic: str, max_results: int = 5) -> str:
    """Search for academic papers on arXiv.

    Args:
        topic: The research topic to search for.
        max_results: Maximum papers to retrieve (1-20, default 5).

    Returns:
        Formatted search results with titles, authors, dates, and links.
    """
    max_results = max(1, min(max_results, 20))
    papers = await search_arxiv(topic, max_results)
    if not papers:
        return f"No papers found for '{topic}'. Check spelling or try broader terms."

    save_papers(topic, papers)

    lines = [f"Found **{len(papers)}** papers on *{topic}*:\n"]
    for i, (pid, p) in enumerate(papers.items(), 1):
        lines.append(f"### {i}. {p['title']}")
        lines.append(f"**Authors:** {', '.join(p['authors'])}")
        lines.append(f"**Published:** {p['published']}")
        lines.append(f"**Categories:** {', '.join(p.get('categories', []))}")
        abstract = p["summary"][:250] + ("..." if len(p["summary"]) > 250 else "")
        lines.append(f"**Abstract:** {abstract}")
        lines.append(f"[PDF]({p['pdf_url']}) | [Abstract]({p.get('abs_url', '')})\n")
    return "\n".join(lines)


@mcp.tool()
def analyze_papers(topic: str) -> str:
    """Generate a comprehensive literature review for a topic.

    Args:
        topic: The research topic (must have been searched first).

    Returns:
        Full literature review in Markdown.
    """
    papers = load_papers(topic)
    if not papers:
        return f"No papers for '{topic}'. Run search_papers first."
    return build_literature_review(topic, papers)


@mcp.tool()
def compare_papers(paper_ids: List[str], comparison_criteria: str = "methodology") -> str:
    """Compare multiple papers across a chosen criterion.

    Args:
        paper_ids: List of arXiv paper IDs to compare.
        comparison_criteria: Focus area -- methodology, results, or applications.

    Returns:
        Comparative analysis in Markdown.
    """
    all_papers: Dict[str, Dict[str, Any]] = {}
    for slug in list_saved_topics():
        all_papers.update(load_papers(slug))

    selected = {pid: all_papers[pid] for pid in paper_ids if pid in all_papers}
    if not selected:
        return f"None of the requested IDs ({', '.join(paper_ids)}) were found."

    focus = comparison_criteria.lower()
    kw_map = {
        "methodology": ["method", "approach", "algorithm", "technique", "framework", "model"],
        "results": ["result", "outcome", "performance", "accuracy", "improvement", "finding"],
        "applications": ["application", "deploy", "implement", "practical", "real-world"],
    }
    keywords = kw_map.get(focus, kw_map["methodology"])

    md = [f"# Paper Comparison -- {comparison_criteria.title()}\n",
          f"**Papers compared:** {len(selected)}\n"]
    for i, (pid, p) in enumerate(selected.items(), 1):
        md.append(f"## {i}. {p['title']}")
        md.append(f"**Authors:** {', '.join(p['authors'])}")
        md.append(f"**Published:** {p['published']}")
        md.append(f"**Categories:** {', '.join(p.get('categories', []))}\n")
        relevant = [
            s.strip() + "."
            for s in p["summary"].split(".")
            if any(k in s.lower() for k in keywords)
        ]
        md.append(f"**{comparison_criteria.title()} excerpt:** {' '.join(relevant[:3]) or p['summary'][:300]}\n---\n")

    all_sums = [p["summary"] for p in selected.values()]
    kws = extract_keywords(all_sums, top_n=8)
    md.append("## Synthesis")
    md.append(f"**Common themes:** {', '.join(k for k, _ in kws[:6])}")
    md.append(f"**Distinct approaches:** {len(selected)} papers with varying focus areas")
    return "\n".join(md)


@mcp.tool()
def track_research_trends(topic: str) -> str:
    """Track research trends over time for a topic.

    Args:
        topic: The research topic (must have been searched first).

    Returns:
        Trend analysis report.
    """
    papers = load_papers(topic)
    if not papers:
        return f"No papers for '{topic}'. Run search_papers first."
    return build_trend_report(topic, papers)


@mcp.tool()
def export_bibtex(topic: str) -> str:
    """Export saved papers as BibTeX references.

    Args:
        topic: The research topic.

    Returns:
        BibTeX-formatted references.
    """
    papers = load_papers(topic)
    if not papers:
        return f"No papers for '{topic}'."
    return papers_to_bibtex(papers)


@mcp.tool()
def export_citations(topic: str, style: str = "apa") -> str:
    """Export saved papers as formatted citations.

    Args:
        topic: The research topic.
        style: Citation style -- apa or table.

    Returns:
        Formatted citations.
    """
    papers = load_papers(topic)
    if not papers:
        return f"No papers for '{topic}'."
    if style == "table":
        return papers_to_markdown_table(papers)
    return papers_to_apa_citations(papers)


# ---- RAG / AI tools ------------------------------------------------------

@mcp.tool()
def semantic_search(query: str, top_k: int = 8) -> str:
    """Search across ALL saved papers using TF-IDF retrieval at chunk level.

    Unlike keyword search, this finds semantically relevant passages
    even if the exact words differ.

    Args:
        query: Natural-language question or topic.
        top_k: Number of chunks to retrieve.

    Returns:
        Ranked list of relevant text passages with paper metadata.
    """
    results = semantic_search_all(query, top_k=top_k)
    if not results:
        return "No relevant passages found. Search for papers first."

    md = [f"## Semantic Search: '{query}'\n"]
    seen: set = set()
    for r in results:
        pid = r.get("paper_id", "")
        if pid not in seen:
            seen.add(pid)
            md.append(f"### {r.get('title', 'Unknown')}")
            md.append(f"*Paper: {pid} | Topic: {r.get('topic', '')}*\n")
        md.append(f"> **[Score {r['score']:.3f}]** {r['text']}\n")
    md.append(f"\n*{len(results)} chunks from {len(seen)} papers.*")
    return "\n".join(md)


@mcp.tool()
def ask_papers(question: str) -> str:
    """RAG-style question answering against the saved paper corpus.

    Retrieves the most relevant chunks and assembles a grounded
    context block that an LLM can use to answer the question.

    Args:
        question: A natural-language research question.

    Returns:
        Retrieved context with source attributions.
    """
    return build_rag_context(question, top_k=6)


@mcp.tool()
def find_related(paper_id: str, topic: str) -> str:
    """Find papers most similar to a given paper using TF-IDF cosine similarity.

    Args:
        paper_id: arXiv ID of the reference paper.
        topic: Topic under which the paper was saved.

    Returns:
        Ranked list of similar papers with similarity scores.
    """
    papers = load_papers(topic)
    if not papers:
        return f"No papers for '{topic}'."
    if paper_id not in papers:
        return f"Paper '{paper_id}' not found in topic '{topic}'."

    similar = find_similar_papers(paper_id, papers, top_k=5)
    if not similar:
        return "Not enough papers to compute similarity."

    ref = papers[paper_id]
    md = [f"## Papers Similar to: {ref['title']}\n"]
    for s in similar:
        md.append(f"- **{s['title']}** (similarity: {s['score']:.3f})")
    return "\n".join(md)


@mcp.tool()
def identify_gaps(topic: str) -> str:
    """Analyse research gaps across 9 dimensions (scalability, privacy, etc.).

    Scans all paper abstracts and classifies each dimension as
    covered, under-explored, or a gap.

    Args:
        topic: The research topic to analyse.

    Returns:
        Gap analysis table with actionable detail.
    """
    papers = load_papers(topic)
    if not papers:
        return f"No papers for '{topic}'. Search first."

    gaps = analyze_research_gaps(papers)
    md = [f"## Research Gap Analysis: {topic.title()}\n"]
    md.append("| Dimension | Status | Detail |")
    md.append("|-----------|--------|--------|")
    for g in gaps:
        icon = {"gap": "RED", "under-explored": "YELLOW", "covered": "GREEN"}.get(g["status"], "?")
        md.append(f"| {g['dimension']} | [{icon}] {g['status']} | {g['detail']} |")
    open_count = sum(1 for g in gaps if g["status"] in ("gap", "under-explored"))
    md.append(f"\n**{open_count} of {len(gaps)} dimensions have gaps or are under-explored.**")
    return "\n".join(md)


@mcp.tool()
def suggest_questions(topic: str) -> str:
    """Auto-generate research questions from gap analysis + keyword trends.

    Args:
        topic: The research topic.

    Returns:
        Numbered list of research questions.
    """
    papers = load_papers(topic)
    if not papers:
        return f"No papers for '{topic}'. Search first."

    questions = generate_research_questions(topic, papers)
    if not questions:
        return "Could not generate questions — not enough data."

    md = [f"## Suggested Research Questions: {topic.title()}\n"]
    for i, q in enumerate(questions, 1):
        md.append(f"{i}. {q}")
    return "\n".join(md)


@mcp.tool()
async def research_agent(topic: str, max_papers: int = 8) -> str:
    """Run a full autonomous research pipeline (multi-step agent).

    Executes: query expansion -> arXiv search -> chunking & indexing ->
    TF-IDF retrieval test -> gap analysis -> question generation ->
    similarity network -> literature review.

    Args:
        topic: The research topic to investigate.
        max_papers: Number of papers to retrieve (1-20).

    Returns:
        Comprehensive multi-step research report.
    """
    return await run_research_agent(topic, max_papers=max(1, min(max_papers, 20)))


@mcp.tool()
def smart_search(query: str, top_k: int = 8) -> str:
    """Hybrid retrieval using TF-IDF + BM25 with Reciprocal Rank Fusion.

    Two-stage pipeline that combines two ranking algorithms for better
    precision than either alone.  Production-grade retrieval pattern.

    Args:
        query: Natural-language search query.
        top_k: Number of results to return.

    Returns:
        Ranked passages with RRF scores.
    """
    results = hybrid_search(query, top_k=top_k)
    if not results:
        return "No results. Search for papers first."

    md = [f"## Hybrid Search: '{query}'\n"]
    md.append("*TF-IDF + BM25 → Reciprocal Rank Fusion*\n")
    seen: set = set()
    for r in results:
        pid = r.get("paper_id", "")
        if pid not in seen:
            seen.add(pid)
            md.append(f"### {r.get('title', 'Unknown')}")
            md.append(f"*Paper: {pid} | RRF: {r.get('rrf_score', 0):.4f}*\n")
        md.append(f"> {r.get('text', '')}\n")
    md.append(f"\n*{len(results)} chunks from {len(seen)} papers.*")
    return "\n".join(md)


@mcp.tool()
def summarize_corpus(topic: str, num_sentences: int = 8) -> str:
    """Generate a multi-document extractive summary using TextRank.

    Builds a sentence-similarity graph and runs PageRank to find
    the most representative sentences across all papers.

    Args:
        topic: The research topic.
        num_sentences: Number of sentences in summary (3-15).

    Returns:
        TextRank extractive summary.
    """
    papers = load_papers(topic)
    if not papers:
        return f"No papers for '{topic}'. Search first."
    return textrank_summarize(papers, num_sentences=max(3, min(num_sentences, 15)))


@mcp.tool()
def extract_named_entities(topic: str) -> str:
    """Extract AI models, datasets, metrics, and tasks from paper abstracts.

    Uses regex-based named entity recognition tuned for ML/AI literature.

    Args:
        topic: The research topic.

    Returns:
        Categorised entity lists with mention counts.
    """
    papers = load_papers(topic)
    if not papers:
        return f"No papers for '{topic}'. Search first."

    entities = extract_entities(papers)
    md = [f"## Named Entities: {topic.title()}\n"]
    for cat in ["models", "datasets", "metrics", "tasks"]:
        items = entities.get(cat, [])
        md.append(f"### {cat.title()} ({len(items)} found)\n")
        if items:
            for e in items[:10]:
                md.append(f"- **{e['name']}** — {e['count']} mention(s) in {len(e['papers'])} paper(s)")
        else:
            md.append("*None detected.*")
        md.append("")
    return "\n".join(md)


@mcp.tool()
def cluster_topic(topic: str, n_clusters: int = 3) -> str:
    """Cluster papers into groups using K-Means on TF-IDF vectors.

    Automatically labels each cluster by its top keywords.

    Args:
        topic: The research topic.
        n_clusters: Number of clusters (2-8).

    Returns:
        Cluster descriptions with paper assignments.
    """
    papers = load_papers(topic)
    if not papers:
        return f"No papers for '{topic}'. Search first."

    clusters = cluster_papers(papers, n_clusters=max(2, min(n_clusters, 8)))
    md = [f"## Paper Clusters: {topic.title()}\n"]
    md.append(f"*K-Means with K={len(clusters)} on TF-IDF vectors*\n")
    for c in clusters:
        md.append(f"### Cluster {c['cluster_id']}: {c['label']}")
        md.append(f"**{c['size']} paper(s)**\n")
        for p in c["papers"]:
            md.append(f"- {p['title']}")
        md.append("")
    return "\n".join(md)


@mcp.tool()
def topic_evolution(topic: str) -> str:
    """Analyse how research focus shifts over time.

    Tracks keyword emergence and decline year-over-year.

    Args:
        topic: The research topic.

    Returns:
        Topic drift report with emerging/declining terms.
    """
    papers = load_papers(topic)
    if not papers:
        return f"No papers for '{topic}'. Search first."
    return format_topic_drift(papers)


@mcp.tool()
def knowledge_graph(topic: str) -> str:
    """Build a concept co-occurrence knowledge graph.

    Extracts key concepts and maps which appear together in papers,
    revealing the conceptual structure of the research field.

    Args:
        topic: The research topic.

    Returns:
        Knowledge graph with nodes (concepts) and edges (co-occurrences).
    """
    papers = load_papers(topic)
    if not papers:
        return f"No papers for '{topic}'. Search first."
    return format_knowledge_graph(papers)


# =========================================================================
# RESOURCES
# =========================================================================

@mcp.resource("papers://folders")
def get_available_folders() -> str:
    """List all available topic folders."""
    topics = list_saved_topics()
    if not topics:
        return "# Available Topics\n\nNo topics found. Search for papers first."
    md = ["# Available Topics\n"]
    for t in topics:
        md.append(f"- {t}")
    return "\n".join(md)


@mcp.resource("papers://{topic}")
def get_topic_papers(topic: str) -> str:
    """Get paper details for a specific topic."""
    papers = load_papers(topic)
    if not papers:
        return f"# No papers for '{topic}'\n\nRun search_papers first."
    return papers_to_markdown_table(papers)


@mcp.resource("research://trends/{topic}")
def get_research_trends(topic: str) -> str:
    """Real-time research trends for a topic."""
    papers = load_papers(topic)
    if not papers:
        return f"# No data for '{topic}'"
    return build_trend_report(topic, papers)


@mcp.resource("research://authors/{topic}")
def get_author_network(topic: str) -> str:
    """Author collaboration network for a topic."""
    papers = load_papers(topic)
    if not papers:
        return f"# No data for '{topic}'"
    return build_author_network(topic, papers)


# =========================================================================
# PROMPTS
# =========================================================================

@mcp.prompt()
def generate_search_prompt(topic: str, num_papers: int = 5) -> str:
    """Generate a prompt for Claude to search and discuss academic papers."""
    return (
        f"Search for {num_papers} academic papers about '{topic}' using "
        f"search_papers(topic='{topic}', max_results={num_papers}).\n\n"
        "For each paper, extract: title, authors, date, key findings, "
        "contributions, methodologies, and relevance.\n\n"
        "Then provide:\n"
        f"- Overview of current research in '{topic}'\n"
        "- Common themes and trends\n"
        "- Research gaps\n"
        "- Most impactful papers\n"
    )


@mcp.prompt()
def generate_literature_review_prompt(topic: str) -> str:
    """Generate a prompt for a comprehensive literature review."""
    return (
        f"Conduct a comprehensive literature review for '{topic}':\n\n"
        f"1. search_papers(topic='{topic}', max_results=10)\n"
        f"2. analyze_papers(topic='{topic}')\n"
        f"3. track_research_trends(topic='{topic}')\n\n"
        "Structure: executive summary, methodology comparison, "
        "key findings, research gaps, future directions, references."
    )


@mcp.prompt()
def generate_comparison_prompt(paper_ids: List[str], focus: str = "methodology") -> str:
    """Prompt to compare specific papers."""
    return (
        f"Compare papers {', '.join(paper_ids)} using "
        f"compare_papers(paper_ids={paper_ids}, comparison_criteria='{focus}').\n\n"
        "Analyse: similarities, differences, strengths, weaknesses, "
        "complementary aspects, recommendations."
    )


@mcp.prompt()
def generate_export_prompt(topic: str) -> str:
    """Prompt to export citations in multiple formats."""
    return (
        f"Export references for '{topic}':\n"
        f"1. export_bibtex(topic='{topic}')\n"
        f"2. export_citations(topic='{topic}', style='apa')\n"
        f"3. export_citations(topic='{topic}', style='table')\n"
    )


# =========================================================================

if __name__ == "__main__":
    logger.info("Starting arXiv MCP Research Server ...")
    mcp.run()
