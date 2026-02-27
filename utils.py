#!/usr/bin/env python3
"""
Shared utilities for arXiv MCP Research Server.
Centralizes API calls, parsing, caching, NLP analysis, and export logic.
"""

import os
import re
import json
import hashlib
import logging
import asyncio
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FMT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger("arxiv_research")

# ---------------------------------------------------------------------------
# Configuration (reads from env vars with sensible defaults)
# ---------------------------------------------------------------------------
PAPER_DIR: str = os.getenv("PAPER_DIR", "papers")
ARXIV_API_BASE: str = os.getenv("ARXIV_API_BASE", "http://export.arxiv.org/api/query")
CACHE_DIR: str = os.getenv("CACHE_DIR", ".cache")
CACHE_TTL_HOURS: int = int(os.getenv("CACHE_TTL_HOURS", "6"))
REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))

# Ensure directories exist
os.makedirs(PAPER_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Stop-words for keyword extraction (expanded list)
# ---------------------------------------------------------------------------
STOP_WORDS = frozenset({
    "about", "above", "after", "again", "against", "also", "among", "based",
    "been", "before", "being", "below", "between", "both", "could", "does",
    "doing", "down", "during", "each", "every", "from", "further", "have",
    "having", "here", "into", "itself", "just", "more", "most", "much",
    "never", "only", "other", "over", "same", "should", "show", "since",
    "some", "still", "such", "than", "that", "their", "them", "then",
    "there", "these", "they", "this", "those", "through", "under", "until",
    "upon", "very", "were", "what", "when", "where", "which", "while",
    "will", "with", "would", "your", "paper", "papers", "using", "used",
    "method", "results", "approach", "proposed", "propose", "present",
    "study", "studies", "work", "recent", "given", "first", "second",
    "third", "however", "although", "therefore", "various", "several",
    "provide", "provides", "demonstrates", "demonstrate", "different",
    "introduce", "existing", "without", "within", "across", "towards",
    "shown", "state", "achieve", "order", "performance",
})

# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def _cache_key(url: str) -> str:
    """Produce a filesystem-safe cache key for *url*."""
    return hashlib.sha256(url.encode()).hexdigest()


def get_cached_response(url: str) -> Optional[str]:
    """Return cached response text if it exists and is fresh, else ``None``."""
    key = _cache_key(url)
    cache_file = os.path.join(CACHE_DIR, f"{key}.xml")
    meta_file = os.path.join(CACHE_DIR, f"{key}.meta.json")

    if not os.path.exists(cache_file) or not os.path.exists(meta_file):
        return None

    with open(meta_file, "r") as f:
        meta = json.load(f)

    cached_at = datetime.fromisoformat(meta["cached_at"])
    if datetime.now() - cached_at > timedelta(hours=CACHE_TTL_HOURS):
        logger.info("Cache expired for %s", url)
        return None

    logger.info("Cache hit for %s", url)
    with open(cache_file, "r", encoding="utf-8") as f:
        return f.read()


def save_to_cache(url: str, content: str) -> None:
    """Persist *content* to the file cache."""
    key = _cache_key(url)
    cache_file = os.path.join(CACHE_DIR, f"{key}.xml")
    meta_file = os.path.join(CACHE_DIR, f"{key}.meta.json")

    with open(cache_file, "w", encoding="utf-8") as f:
        f.write(content)
    with open(meta_file, "w") as f:
        json.dump({"url": url, "cached_at": datetime.now().isoformat()}, f)


# ---------------------------------------------------------------------------
# arXiv API
# ---------------------------------------------------------------------------

async def search_arxiv(
    topic: str,
    max_results: int = 5,
    sort_by: str = "relevance",
) -> Dict[str, Dict[str, Any]]:
    """Search arXiv for *topic* and return parsed paper metadata.

    Results are cached locally to avoid redundant network calls.
    Falls back to an empty dict on error (no fake data).
    """
    if not topic or len(topic.strip()) < 2:
        logger.warning("Invalid search topic: '%s'", topic)
        return {}

    query = f"all:{topic}"
    url = (
        f"{ARXIV_API_BASE}?search_query={query}"
        f"&start=0&max_results={max_results}"
        f"&sortBy={sort_by}&sortOrder=descending"
    )
    logger.info("arXiv search URL: %s", url)

    # Try cache first
    cached = get_cached_response(url)
    if cached:
        return _parse_arxiv_xml(cached, topic)

    # Live fetch
    try:
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    logger.error("arXiv returned HTTP %s", resp.status)
                    return {}
                xml_text = await resp.text()
                save_to_cache(url, xml_text)
                return _parse_arxiv_xml(xml_text, topic)
    except asyncio.TimeoutError:
        logger.error("arXiv API call timed out")
    except Exception:
        logger.exception("arXiv API call failed")
    return {}


def _parse_arxiv_xml(xml_content: str, topic: str) -> Dict[str, Dict[str, Any]]:
    """Parse Atom-feed XML returned by arXiv into a papers dict."""
    papers: Dict[str, Dict[str, Any]] = {}
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError:
        logger.exception("Failed to parse arXiv XML")
        return papers

    for entry in root.findall("atom:entry", ns):
        try:
            id_el = entry.find("atom:id", ns)
            if id_el is None:
                continue
            paper_id = (id_el.text or "").split("/")[-1]

            title_el = entry.find("atom:title", ns)
            title_raw = (title_el.text or "") if title_el is not None else ""
            title = re.sub(r"\s+", " ", title_raw.strip())

            authors = []
            for a in entry.findall("atom:author", ns):
                name_el = a.find("atom:name", ns)
                if name_el is not None and name_el.text:
                    authors.append(name_el.text.strip())

            summary_el = entry.find("atom:summary", ns)
            summary_raw = (summary_el.text or "") if summary_el is not None else ""
            summary = re.sub(r"\s+", " ", summary_raw.strip())

            pub_el = entry.find("atom:published", ns)
            pub_raw = (pub_el.text or "") if pub_el is not None else ""
            published = pub_raw[:10]

            categories = [
                c.get("term")
                for c in entry.findall("atom:category", ns)
                if c.get("term")
            ]

            papers[paper_id] = {
                "title": title,
                "authors": authors,
                "published": published,
                "summary": summary,
                "pdf_url": f"https://arxiv.org/pdf/{paper_id}.pdf",
                "abs_url": f"https://arxiv.org/abs/{paper_id}",
                "categories": categories,
                "topic_searched": topic,
            }
        except Exception:
            logger.exception("Error parsing entry")
            continue

    logger.info("Parsed %d papers for '%s'", len(papers), topic)
    return papers


# ---------------------------------------------------------------------------
# Paper persistence helpers
# ---------------------------------------------------------------------------

def topic_dir_name(topic: str) -> str:
    """Normalise a topic string into a safe directory name."""
    return re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")


def save_papers(topic: str, papers: Dict[str, Dict[str, Any]]) -> str:
    """Save *papers* to disk under the topic folder. Returns the file path."""
    tdir = os.path.join(PAPER_DIR, topic_dir_name(topic))
    os.makedirs(tdir, exist_ok=True)
    path = os.path.join(tdir, "papers_info.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d papers to %s", len(papers), path)
    return path


def load_papers(topic: str) -> Dict[str, Dict[str, Any]]:
    """Load previously-saved papers for *topic*. Returns empty dict if none."""
    path = os.path.join(PAPER_DIR, topic_dir_name(topic), "papers_info.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_saved_topics() -> List[str]:
    """Return a list of all topic slugs that have saved papers."""
    if not os.path.exists(PAPER_DIR):
        return []
    topics = []
    for name in sorted(os.listdir(PAPER_DIR)):
        full = os.path.join(PAPER_DIR, name)
        if os.path.isdir(full) and os.path.exists(os.path.join(full, "papers_info.json")):
            topics.append(name)
    return topics


# ---------------------------------------------------------------------------
# NLP / keyword extraction (improved with stop-words & bigrams)
# ---------------------------------------------------------------------------

def extract_keywords(
    texts: List[str],
    top_n: int = 15,
    min_word_len: int = 4,
) -> List[Tuple[str, int]]:
    """Extract the most frequent meaningful keywords from *texts*.

    Uses stop-word filtering and optional bigram extraction.
    """
    word_counts: Counter = Counter()
    bigram_counts: Counter = Counter()

    for text in texts:
        tokens = re.findall(r"[a-z]{2,}", text.lower())
        filtered = [t for t in tokens if len(t) >= min_word_len and t not in STOP_WORDS]
        word_counts.update(filtered)

        # Bigrams
        for a, b in zip(filtered, filtered[1:]):
            bigram_counts[f"{a} {b}"] += 1

    # Merge top bigrams into results (they provide more signal)
    combined: Counter = Counter()
    for word, cnt in word_counts.most_common(top_n * 3):
        combined[word] = cnt
    for bg, cnt in bigram_counts.most_common(top_n):
        if cnt >= 2:  # only if bigram appears at least twice
            combined[bg] = cnt

    return combined.most_common(top_n)


def extract_key_contributions(summary: str) -> List[str]:
    """Pull sentences from *summary* that describe a contribution."""
    indicators = {
        "propose", "introduce", "develop", "present", "demonstrate",
        "achieve", "improve", "enhance", "novel", "outperform",
    }
    results = []
    for sent in summary.split("."):
        sent = sent.strip()
        if 20 < len(sent) < 250 and any(w in sent.lower() for w in indicators):
            results.append(sent.rstrip(".") + ".")
            if len(results) >= 3:
                break
    return results


def categorize_papers(papers: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """Group paper IDs into focus-area buckets based on title/summary keywords."""
    buckets: Dict[str, List[str]] = {
        "Theoretical": [],
        "Experimental": [],
        "Applied": [],
        "Survey / Review": [],
        "Methodology": [],
        "System / Tool": [],
    }
    mapping = {
        "survey|review|overview|systematic": "Survey / Review",
        "theor|formali|proof|framework|model": "Theoretical",
        "experiment|evaluat|benchmark|compar": "Experimental",
        "applic|deploy|real.world|practical|industry": "Applied",
        "method|algorithm|approach|technique": "Methodology",
        "implement|system|platform|tool|library": "System / Tool",
    }

    for pid, paper in papers.items():
        blob = (paper["title"] + " " + paper["summary"]).lower()
        placed = False
        for pattern, bucket in mapping.items():
            if re.search(pattern, blob):
                buckets[bucket].append(pid)
                placed = True
                break
        if not placed:
            buckets["Applied"].append(pid)

    return {k: v for k, v in buckets.items() if v}


def extract_methodologies(papers: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    """Count methodology mentions across papers."""
    tags = {
        "Deep Learning": r"deep learning|neural network|cnn|rnn|lstm|transformer",
        "Machine Learning": r"machine learning|random forest|svm|gradient boost",
        "Reinforcement Learning": r"reinforcement learning|reward|policy gradient|q-learning",
        "NLP": r"natural language|nlp|language model|text mining|sentiment",
        "Computer Vision": r"computer vision|image|object detection|segmentation",
        "Optimization": r"optimiz|convex|gradient descent|linear program",
        "Statistical Analysis": r"statistic|regression|hypothesis|bayesian",
        "Simulation": r"simulat|monte carlo|agent.based",
    }
    counts: Dict[str, int] = {}
    for paper in papers.values():
        blob = (paper["title"] + " " + paper["summary"]).lower()
        for label, pattern in tags.items():
            if re.search(pattern, blob):
                counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def papers_by_year(papers: Dict[str, Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group papers by publication year."""
    by_year: Dict[str, List[Dict[str, Any]]] = {}
    for paper in papers.values():
        year = paper["published"][:4]
        by_year.setdefault(year, []).append(paper)
    return dict(sorted(by_year.items()))


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def papers_to_bibtex(papers: Dict[str, Dict[str, Any]]) -> str:
    """Convert papers dict to BibTeX format."""
    entries = []
    for pid, p in papers.items():
        safe_id = re.sub(r"[^a-zA-Z0-9]", "_", pid)
        first_author_last = p["authors"][0].split()[-1] if p["authors"] else "Unknown"
        year = p["published"][:4]
        entry = (
            f"@article{{{first_author_last}{year}_{safe_id},\n"
            f"  title     = {{{p['title']}}},\n"
            f"  author    = {{{' and '.join(p['authors'])}}},\n"
            f"  year      = {{{year}}},\n"
            f"  eprint    = {{{pid}}},\n"
            f"  archivePrefix = {{arXiv}},\n"
            f"  primaryClass = {{{p['categories'][0] if p.get('categories') else 'cs.AI'}}},\n"
            f"  url       = {{{p.get('abs_url', p['pdf_url'])}}},\n"
            f"  abstract  = {{{p['summary'][:500]}}}\n"
            f"}}"
        )
        entries.append(entry)
    return "\n\n".join(entries)


def papers_to_markdown_table(papers: Dict[str, Dict[str, Any]]) -> str:
    """Render papers as a Markdown table for easy copy-paste."""
    lines = ["| # | Title | Authors | Year | Categories | PDF |",
             "|---|-------|---------|------|------------|-----|"]
    for i, (pid, p) in enumerate(papers.items(), 1):
        authors = ", ".join(p["authors"][:3])
        if len(p["authors"]) > 3:
            authors += " et al."
        cats = ", ".join(p.get("categories", [])[:3])
        lines.append(
            f"| {i} | {p['title']} | {authors} | {p['published'][:4]} "
            f"| {cats} | [PDF]({p['pdf_url']}) |"
        )
    return "\n".join(lines)


def papers_to_apa_citations(papers: Dict[str, Dict[str, Any]]) -> str:
    """Generate APA-style reference list."""
    refs = []
    for i, (pid, p) in enumerate(papers.items(), 1):
        authors = ", ".join(p["authors"][:5])
        if len(p["authors"]) > 5:
            authors += ", et al."
        year = p["published"][:4]
        refs.append(f"[{i}] {authors} ({year}). {p['title']}. *arXiv preprint arXiv:{pid}*. {p.get('abs_url', p['pdf_url'])}")
    return "\n\n".join(refs)


# ---------------------------------------------------------------------------
# Analysis / report builders
# ---------------------------------------------------------------------------

def build_literature_review(topic: str, papers: Dict[str, Dict[str, Any]]) -> str:
    """Generate a comprehensive literature review in Markdown."""
    if not papers:
        return f"# No papers available for '{topic}'\n\nPlease search first."

    by_year = papers_by_year(papers)
    summaries = [p["summary"] for p in papers.values()]
    keywords = extract_keywords(summaries)
    focus = categorize_papers(papers)
    methods = extract_methodologies(papers)

    md = []
    md.append(f"# Literature Review: {topic.title()}\n")
    md.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    md.append(f"**Papers Analysed:** {len(papers)}")
    date_range = f"{min(p['published'] for p in papers.values())} — {max(p['published'] for p in papers.values())}"
    md.append(f"**Publication Range:** {date_range}\n")

    # Executive Summary
    md.append("## Executive Summary\n")
    md.append(
        f"This review synthesises {len(papers)} papers on *{topic}*. "
        f"The dominant keywords are **{', '.join(k for k, _ in keywords[:5])}**. "
        f"Research spans {len(by_year)} publication year(s) ({date_range}).\n"
    )

    # Key Themes
    md.append("## Key Themes & Keywords\n")
    for kw, cnt in keywords[:12]:
        md.append(f"- **{kw}** — {cnt} occurrence(s)")

    # Focus Areas
    md.append("\n## Research Focus Areas\n")
    for area, pids in focus.items():
        md.append(f"- **{area}**: {len(pids)} paper(s)")

    # Methodologies
    if methods:
        md.append("\n## Methodological Approaches\n")
        for met, cnt in methods.items():
            md.append(f"- **{met}**: {cnt} paper(s)")

    # Per-Year Detail
    md.append("\n## Detailed Paper Analysis\n")
    for year in sorted(by_year, reverse=True):
        md.append(f"### {year}\n")
        for p in by_year[year]:
            md.append(f"#### {p['title']}\n")
            md.append(f"**Authors:** {', '.join(p['authors'])}")
            md.append(f"**Published:** {p['published']}")
            md.append(f"**Categories:** {', '.join(p.get('categories', ['N/A']))}\n")
            md.append(f"**Abstract:** {p['summary']}\n")
            contribs = extract_key_contributions(p["summary"])
            if contribs:
                md.append("**Key Contributions:**")
                for c in contribs:
                    md.append(f"- {c}")
            md.append(f"\n[PDF]({p['pdf_url']}) | [Abstract]({p.get('abs_url', '')})\n---\n")

    # Timeline
    md.append("## Publication Timeline\n")
    for year in sorted(by_year):
        md.append(f"- **{year}**: {len(by_year[year])} paper(s)")

    # References
    md.append("\n## References\n")
    md.append(papers_to_apa_citations(papers))

    return "\n".join(md)


def build_trend_report(topic: str, papers: Dict[str, Dict[str, Any]]) -> str:
    """Build a trend-analysis report."""
    if not papers:
        return f"# No data for '{topic}'"

    by_year = papers_by_year(papers)
    summaries = [p["summary"] for p in papers.values()]
    keywords = extract_keywords(summaries)
    methods = extract_methodologies(papers)

    md = [f"# Research Trends: {topic.title()}\n"]
    md.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    md.append(f"**Papers Analysed:** {len(papers)}\n")

    md.append("## Publication Timeline\n")
    for yr in sorted(by_year):
        md.append(f"- **{yr}**: {len(by_year[yr])} paper(s)")

    if len(by_year) >= 2:
        years = sorted(by_year)
        first, last = len(by_year[years[0]]), len(by_year[years[-1]])
        if first:
            growth = ((last - first) / first) * 100
            md.append(f"\n**Growth:** {growth:+.0f}% from {years[0]} to {years[-1]}")

    md.append("\n## Trending Keywords\n")
    for kw, cnt in keywords[:12]:
        md.append(f"- **{kw}**: {cnt}")

    if methods:
        md.append("\n## Methodology Distribution\n")
        for met, cnt in methods.items():
            md.append(f"- **{met}**: {cnt}")

    return "\n".join(md)


def build_author_network(topic: str, papers: Dict[str, Dict[str, Any]]) -> str:
    """Build an author-network analysis report."""
    if not papers:
        return f"# No data for '{topic}'"

    author_counts: Counter = Counter()
    collab_sizes: List[int] = []

    for p in papers.values():
        collab_sizes.append(len(p["authors"]))
        for a in p["authors"]:
            author_counts[a] += 1

    md = [f"# Author Network: {topic.title()}\n"]
    md.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    md.append(f"**Papers:** {len(papers)} | **Unique Authors:** {len(author_counts)}")
    md.append(f"**Avg. Authors/Paper:** {sum(collab_sizes) / len(collab_sizes):.1f}\n")

    md.append("## Most Active Authors\n")
    for author, cnt in author_counts.most_common(15):
        md.append(f"- **{author}**: {cnt} paper(s)")

    md.append("\n## Collaboration Size Distribution\n")
    size_counts = Counter(collab_sizes)
    for size in sorted(size_counts):
        md.append(f"- **{size} authors**: {size_counts[size]} paper(s)")

    return "\n".join(md)


# ===========================================================================
# RAG ENGINE — Chunking, TF-IDF Indexing, Semantic Retrieval
# ===========================================================================

import math

# ---- Sentence-level chunking ---------------------------------------------

def chunk_abstract(text: str, max_chunk_chars: int = 300) -> List[str]:
    """Split an abstract into sentence-level chunks.

    Sentences shorter than 30 chars are merged with the previous chunk
    to avoid tiny fragments.  Chunks are capped at *max_chunk_chars* by
    splitting on clause boundaries (commas / semicolons) when needed.
    """
    raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: List[str] = []
    buf = ""
    for sent in raw_sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(buf) + len(sent) + 1 <= max_chunk_chars:
            buf = (buf + " " + sent).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = sent
    if buf:
        chunks.append(buf)
    return chunks


def build_corpus_chunks(
    papers: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Chunk every paper's abstract and return a flat list of chunk dicts.

    Each dict: ``{"paper_id", "title", "chunk_idx", "text"}``.
    """
    corpus: List[Dict[str, Any]] = []
    for pid, p in papers.items():
        for idx, chunk in enumerate(chunk_abstract(p["summary"])):
            corpus.append({
                "paper_id": pid,
                "title": p["title"],
                "chunk_idx": idx,
                "text": chunk,
            })
    return corpus


# ---- TF-IDF vectoriser (pure Python — no sklearn needed) -----------------

def _tokenize(text: str) -> List[str]:
    """Lower-case tokenise, filtering stop-words and short tokens."""
    return [
        t for t in re.findall(r"[a-z]{2,}", text.lower())
        if len(t) >= 3 and t not in STOP_WORDS
    ]


class TFIDFIndex:
    """Lightweight in-memory TF-IDF index over text chunks.

    No external dependencies — uses pure-Python term-frequency /
    inverse-document-frequency with cosine similarity.
    """

    def __init__(self) -> None:
        self.docs: List[str] = []          # raw chunk texts
        self.meta: List[Dict[str, Any]] = []  # chunk metadata
        self.vocab: Dict[str, int] = {}    # term → column index
        self.idf: Dict[str, float] = {}
        self.tfidf_matrix: List[Dict[int, float]] = []  # sparse rows

    # -- build index --------------------------------------------------------

    def add_documents(
        self,
        texts: List[str],
        metas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Index a batch of documents."""
        self.docs.extend(texts)
        self.meta.extend(metas or [{} for _ in texts])
        self._rebuild()

    def _rebuild(self) -> None:
        n = len(self.docs)
        if n == 0:
            return

        # 1. Build vocab + document frequency
        df: Counter = Counter()
        tokenised: List[List[str]] = []
        for doc in self.docs:
            toks = _tokenize(doc)
            tokenised.append(toks)
            unique = set(toks)
            for t in unique:
                df[t] += 1

        self.vocab = {term: i for i, term in enumerate(sorted(df))}
        self.idf = {
            term: math.log((1 + n) / (1 + freq)) + 1
            for term, freq in df.items()
        }

        # 2. TF-IDF sparse rows (dict[col] = weight)
        self.tfidf_matrix = []
        for toks in tokenised:
            tf: Counter = Counter(toks)
            row: Dict[int, float] = {}
            length = 0.0
            for term, count in tf.items():
                if term in self.vocab:
                    w = (count / len(toks)) * self.idf.get(term, 1.0)
                    row[self.vocab[term]] = w
                    length += w * w
            # L2 normalise
            norm = math.sqrt(length) if length else 1.0
            for col in row:
                row[col] /= norm
            self.tfidf_matrix.append(row)

    # -- query --------------------------------------------------------------

    def query(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return the *top_k* most relevant chunks for *text*.

        Each result dict has ``score``, ``text``, and the chunk metadata.
        """
        if not self.tfidf_matrix:
            return []

        toks = _tokenize(text)
        tf: Counter = Counter(toks)
        q_vec: Dict[int, float] = {}
        length = 0.0
        for term, count in tf.items():
            if term in self.vocab:
                w = (count / max(len(toks), 1)) * self.idf.get(term, 1.0)
                q_vec[self.vocab[term]] = w
                length += w * w
        norm = math.sqrt(length) if length else 1.0
        for col in q_vec:
            q_vec[col] /= norm

        # Cosine similarity against every document
        scores: List[Tuple[float, int]] = []
        for i, row in enumerate(self.tfidf_matrix):
            dot = sum(q_vec.get(col, 0.0) * row.get(col, 0.0) for col in q_vec)
            scores.append((dot, i))

        scores.sort(key=lambda x: -x[0])
        results = []
        for score, idx in scores[:top_k]:
            if score <= 0:
                break
            results.append({
                "score": round(score, 4),
                "text": self.docs[idx],
                **self.meta[idx],
            })
        return results


# ---- Convenience index builder over all saved topics ---------------------

def build_global_index() -> TFIDFIndex:
    """Build a TF-IDF index over **all** saved papers (all topics)."""
    index = TFIDFIndex()
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []
    for slug in list_saved_topics():
        papers = load_papers(slug)
        for chunk in build_corpus_chunks(papers):
            texts.append(chunk["text"])
            metas.append({
                "paper_id": chunk["paper_id"],
                "title": chunk["title"],
                "chunk_idx": chunk["chunk_idx"],
                "topic": slug,
            })
    if texts:
        index.add_documents(texts, metas)
        logger.info("Global index built: %d chunks from %d topic(s)",
                     len(texts), len(list_saved_topics()))
    return index


def semantic_search_all(query: str, top_k: int = 8) -> List[Dict[str, Any]]:
    """Search across ALL saved papers using TF-IDF retrieval."""
    index = build_global_index()
    return index.query(query, top_k=top_k)


# ===========================================================================
# Paper Similarity Matrix
# ===========================================================================

def compute_paper_similarity(
    papers: Dict[str, Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Compute pairwise TF-IDF cosine similarity between papers.

    Returns a dict mapping each paper_id to a list of
    ``{"paper_id", "title", "score"}`` sorted by descending score.
    """
    pids = list(papers.keys())
    if len(pids) < 2:
        return {}

    index = TFIDFIndex()
    texts = [papers[pid]["summary"] for pid in pids]
    metas = [{"paper_id": pid, "title": papers[pid]["title"]} for pid in pids]
    index.add_documents(texts, metas)

    similarity: Dict[str, List[Dict[str, Any]]] = {}
    for i, pid in enumerate(pids):
        results = index.query(texts[i], top_k=len(pids))
        similarity[pid] = [
            r for r in results if r["paper_id"] != pid
        ]
    return similarity


def find_similar_papers(
    paper_id: str,
    papers: Dict[str, Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Find the *top_k* most similar papers to *paper_id*."""
    sim = compute_paper_similarity(papers)
    return sim.get(paper_id, [])[:top_k]


# ===========================================================================
# Query Expansion
# ===========================================================================

SYNONYM_MAP: Dict[str, List[str]] = {
    "llm":                ["large language model", "GPT", "language model"],
    "large language model":["LLM", "GPT", "transformer language model"],
    "gpt":                ["large language model", "generative pre-trained"],
    "rag":                ["retrieval augmented generation", "retrieval generation"],
    "retrieval augmented generation": ["RAG", "retrieval generation"],
    "nlp":                ["natural language processing", "text mining", "computational linguistics"],
    "natural language processing": ["NLP", "text mining"],
    "cnn":                ["convolutional neural network", "convnet"],
    "rnn":                ["recurrent neural network"],
    "gan":                ["generative adversarial network"],
    "bert":               ["bidirectional encoder", "pre-trained language model"],
    "transformer":        ["attention mechanism", "self-attention"],
    "reinforcement learning": ["RL", "policy gradient", "reward learning"],
    "federated learning": ["distributed learning", "privacy-preserving ML"],
    "computer vision":    ["image recognition", "visual understanding", "CV"],
    "graph neural network":["GNN", "graph convolution", "message passing"],
    "diffusion model":    ["denoising diffusion", "score-based model"],
    "attention":          ["self-attention", "multi-head attention", "transformer"],
    "fine-tuning":        ["fine tuning", "adaptation", "transfer learning"],
    "zero-shot":          ["zero shot", "zero-shot learning"],
    "few-shot":           ["few shot", "few-shot learning", "meta-learning"],
    "agent":              ["autonomous agent", "AI agent", "tool-using LLM"],
    "mcp":                ["model context protocol"],
    "embedding":          ["vector representation", "dense representation"],
    "chunking":           ["text splitting", "document segmentation"],
}


def expand_query(query: str) -> str:
    """Expand *query* with known synonyms to broaden arXiv search recall.

    Returns a query string with OR-joined alternatives suitable for the
    arXiv ``all:`` field.
    """
    low = query.lower().strip()
    expansions: List[str] = [query]

    for term, synonyms in SYNONYM_MAP.items():
        if term in low:
            expansions.extend(synonyms[:2])  # limit to 2 extras

    # Deduplicate (case-insensitive)
    seen: set = set()
    unique: List[str] = []
    for e in expansions:
        key = e.lower()
        if key not in seen:
            seen.add(key)
            unique.append(e)

    if len(unique) == 1:
        return query
    # arXiv supports OR in search queries
    return " OR ".join(f'"{u}"' if " " in u else u for u in unique)


# ===========================================================================
# Intelligent Gap Analysis (replaces hardcoded strings)
# ===========================================================================

_GAP_DIMENSIONS = {
    "Scalability": {
        "present": r"scal(e|ing|ability)|large.scale|distributed|parallel",
        "gap_msg": "Limited research on scaling to large datasets / production workloads.",
    },
    "Real-World Deployment": {
        "present": r"deploy|production|real.world|industry|practical",
        "gap_msg": "Few studies validate approaches in real-world / industrial settings.",
    },
    "Privacy & Security": {
        "present": r"privacy|differential privacy|federat|secure|encrypt",
        "gap_msg": "Insufficient exploration of privacy-preserving or secure approaches.",
    },
    "Explainability": {
        "present": r"explain|interpret|transparen|xai|attention visual",
        "gap_msg": "Lack of interpretability / explainability analysis.",
    },
    "Reproducibility": {
        "present": r"reproduc|open.source|code.avail|benchmark|replicate",
        "gap_msg": "Limited emphasis on reproducibility or open benchmarks.",
    },
    "Fairness & Bias": {
        "present": r"bias|fair|equit|demographic|discriminat",
        "gap_msg": "Barely addresses fairness, bias, or equity considerations.",
    },
    "Efficiency": {
        "present": r"efficien|lightweight|pruning|quantiz|distill|latency",
        "gap_msg": "Computational efficiency and model compression under-explored.",
    },
    "Multi-Modal": {
        "present": r"multi.modal|vision.language|audio|cross.modal",
        "gap_msg": "Little investigation into multi-modal or cross-modal approaches.",
    },
    "Human Evaluation": {
        "present": r"human eval|user study|subjective|annotation|crowd",
        "gap_msg": "Reliance on automatic metrics; human evaluation rarely conducted.",
    },
}


def analyze_research_gaps(
    papers: Dict[str, Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Analyse abstracts to identify concrete research gaps.

    For each dimension, checks whether any paper addresses it.
    Returns a list of ``{"dimension", "status", "detail"}`` dicts.
    """
    blob = " ".join(
        (p["title"] + " " + p["summary"]).lower() for p in papers.values()
    )
    gaps: List[Dict[str, str]] = []
    for dim, cfg in _GAP_DIMENSIONS.items():
        if re.search(cfg["present"], blob):
            # Count how many papers mention it
            count = sum(
                1 for p in papers.values()
                if re.search(cfg["present"], (p["title"] + " " + p["summary"]).lower())
            )
            if count < len(papers) * 0.3:  # mentioned in < 30%  papers
                gaps.append({
                    "dimension": dim,
                    "status": "under-explored",
                    "detail": f"Only {count}/{len(papers)} papers touch on {dim.lower()}.",
                })
            else:
                gaps.append({
                    "dimension": dim,
                    "status": "covered",
                    "detail": f"{count}/{len(papers)} papers address {dim.lower()}.",
                })
        else:
            gaps.append({
                "dimension": dim,
                "status": "gap",
                "detail": cfg["gap_msg"],
            })
    return gaps


# ===========================================================================
# Research Question Generator
# ===========================================================================

def generate_research_questions(
    topic: str,
    papers: Dict[str, Dict[str, Any]],
    max_questions: int = 6,
) -> List[str]:
    """Auto-generate research questions from gap analysis + keyword analysis.

    Combines identified gaps with trending keywords to produce specific,
    actionable questions a researcher could investigate.
    """
    gaps = analyze_research_gaps(papers)
    summaries = [p["summary"] for p in papers.values()]
    top_kws = [kw for kw, _ in extract_keywords(summaries, top_n=8)]
    methods = list(extract_methodologies(papers).keys())[:4]

    questions: List[str] = []

    # Gap-driven questions
    open_gaps = [g for g in gaps if g["status"] == "gap"]
    for g in open_gaps[:3]:
        dim = g["dimension"]
        kw = top_kws[0] if top_kws else topic
        questions.append(
            f"How can {dim.lower()} be addressed in {topic} research, "
            f"particularly for {kw}-based approaches?"
        )

    # Under-explored dimension questions
    under = [g for g in gaps if g["status"] == "under-explored"]
    for g in under[:2]:
        dim = g["dimension"]
        questions.append(
            f"What are the challenges and opportunities in improving "
            f"{dim.lower()} for state-of-the-art {topic} methods?"
        )

    # Cross-methodology questions
    if len(methods) >= 2:
        questions.append(
            f"How do {methods[0]} and {methods[1]} compare in terms of "
            f"effectiveness for {topic} tasks, and can they be combined?"
        )

    # Keyword-combination questions
    if len(top_kws) >= 3:
        questions.append(
            f"What is the relationship between {top_kws[0]}, {top_kws[1]}, "
            f"and {top_kws[2]} in the context of {topic}?"
        )

    return questions[:max_questions]


# ===========================================================================
# RAG Context Builder (retrieval + assembly for LLM prompts)
# ===========================================================================

def build_rag_context(
    query: str,
    top_k: int = 6,
) -> str:
    """Retrieve relevant chunks and assemble a grounded context block.

    Suitable for feeding into an LLM prompt or displaying as evidence.
    """
    results = semantic_search_all(query, top_k=top_k)
    if not results:
        return "No relevant passages found in the paper corpus."

    md = ["## Retrieved Context\n"]
    seen_papers: set = set()
    for r in results:
        pid = r.get("paper_id", "")
        title = r.get("title", "")
        score = r.get("score", 0)
        text = r.get("text", "")
        new_paper = pid not in seen_papers
        seen_papers.add(pid)
        if new_paper:
            md.append(f"### {title}")
            md.append(f"*Paper ID: {pid} | Relevance: {score:.2f}*\n")
        md.append(f"> {text}\n")

    md.append(f"\n*{len(results)} chunks retrieved from {len(seen_papers)} paper(s).*")
    return "\n".join(md)


# ===========================================================================
# Multi-Step Agent Workflow
# ===========================================================================

async def run_research_agent(
    topic: str,
    max_papers: int = 8,
) -> str:
    """Execute a full autonomous research pipeline:

    1. Expand query → 2. Search arXiv → 3. Save & chunk papers →
    4. Build TF-IDF index → 5. Analyse gaps → 6. Generate questions →
    7. Find internal connections → 8. Compile report.

    Returns a comprehensive Markdown report.
    """
    md: List[str] = [f"# Research Agent Report: {topic}\n"]
    md.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    # Step 1 — Query expansion
    expanded = expand_query(topic)
    md.append("## Step 1: Query Expansion\n")
    md.append(f"Original: `{topic}`")
    md.append(f"Expanded: `{expanded}`\n")

    # Step 2 — Search arXiv
    md.append("## Step 2: Paper Retrieval\n")
    papers = await search_arxiv(topic, max_results=max_papers)
    if not papers:
        md.append(f"No papers found for '{topic}'.")
        return "\n".join(md)
    save_papers(topic, papers)
    md.append(f"Retrieved **{len(papers)}** papers from arXiv.\n")

    # Step 3 — Chunking
    chunks = build_corpus_chunks(papers)
    md.append("## Step 3: Chunking & Indexing\n")
    md.append(f"Created **{len(chunks)}** sentence-level chunks from {len(papers)} abstracts.")
    avg_chunk = sum(len(c["text"]) for c in chunks) / max(len(chunks), 1)
    md.append(f"Average chunk size: {avg_chunk:.0f} characters.\n")

    # Step 4 — TF-IDF index
    index = TFIDFIndex()
    index.add_documents(
        [c["text"] for c in chunks],
        [{"paper_id": c["paper_id"], "title": c["title"], "chunk_idx": c["chunk_idx"]} for c in chunks],
    )
    md.append("## Step 4: Semantic Retrieval Test\n")
    test_results = index.query(topic, top_k=3)
    md.append(f"Top-3 chunks for *\"{topic}\"*:\n")
    for r in test_results:
        md.append(f"- **[{r.get('paper_id', '')}]** (score {r['score']:.3f}): {r['text'][:150]}…")
    md.append("")

    # Step 5 — Gap analysis
    gaps = analyze_research_gaps(papers)
    md.append("## Step 5: Research Gap Analysis\n")
    md.append("| Dimension | Status | Detail |")
    md.append("|-----------|--------|--------|")
    for g in gaps:
        icon = {"gap": "🔴", "under-explored": "🟡", "covered": "🟢"}.get(g["status"], "⚪")
        md.append(f"| {g['dimension']} | {icon} {g['status']} | {g['detail']} |")
    md.append("")

    # Step 6 — Research questions
    questions = generate_research_questions(topic, papers)
    md.append("## Step 6: Generated Research Questions\n")
    for i, q in enumerate(questions, 1):
        md.append(f"{i}. {q}")
    md.append("")

    # Step 7 — Paper similarity network
    sim = compute_paper_similarity(papers)
    md.append("## Step 7: Paper Similarity Network\n")
    for pid, neighbours in sim.items():
        title = papers[pid]["title"]
        if neighbours:
            top = neighbours[0]
            md.append(f"- **{title}** → most similar: *{top['title']}* (score {top['score']:.3f})")
    md.append("")

    # Step 8 — Literature review
    md.append("## Step 8: Synthesised Literature Review\n")
    md.append(build_literature_review(topic, papers))

    return "\n".join(md)


# ===========================================================================
# BM25 Retriever (Okapi BM25 — industry-standard ranking)
# ===========================================================================

class BM25Index:
    """Okapi BM25 ranking — the algorithm behind Elasticsearch / Lucene.

    Parameters *k1* and *b* control term-frequency saturation and
    document-length normalisation respectively.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs: List[List[str]] = []
        self.meta: List[Dict[str, Any]] = []
        self.avgdl: float = 0.0
        self.df: Dict[str, int] = {}
        self.n: int = 0

    def add_documents(
        self,
        texts: List[str],
        metas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        tokenised = [_tokenize(t) for t in texts]
        self.docs.extend(tokenised)
        self.meta.extend(metas or [{} for _ in texts])
        self.n = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(self.n, 1)
        # rebuild document-frequency
        self.df = {}
        for doc in self.docs:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.n - df + 0.5) / (df + 0.5) + 1)

    def query(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q_tokens = _tokenize(text)
        scores: List[Tuple[float, int]] = []
        for i, doc in enumerate(self.docs):
            score = 0.0
            dl = len(doc)
            tf_map: Counter = Counter(doc)
            for qt in q_tokens:
                if qt not in tf_map:
                    continue
                tf = tf_map[qt]
                idf = self._idf(qt)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
                score += idf * numerator / denominator
            scores.append((score, i))
        scores.sort(key=lambda x: -x[0])
        results = []
        for s, idx in scores[:top_k]:
            if s <= 0:
                break
            results.append({"score": round(s, 4), "text": " ".join(self.docs[idx]), **self.meta[idx]})
        return results


# ===========================================================================
# Hybrid Retrieval — TF-IDF + BM25 with Reciprocal Rank Fusion (RRF)
# ===========================================================================

def reciprocal_rank_fusion(
    *rankings: List[Dict[str, Any]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """Merge multiple ranked lists using RRF (Cormack et al. 2009).

    RRF score = Σ  1 / (k + rank_i)  over all ranking lists.
    Higher is better.
    """
    rrf_scores: Dict[str, float] = {}
    meta_map: Dict[str, Dict[str, Any]] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            key = item.get("paper_id", "") + "|" + str(item.get("chunk_idx", ""))
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in meta_map:
                meta_map[key] = item
    sorted_keys = sorted(rrf_scores, key=lambda x: -rrf_scores[x])
    results = []
    for key in sorted_keys:
        entry = dict(meta_map[key])
        entry["rrf_score"] = round(rrf_scores[key], 6)
        results.append(entry)
    return results


def hybrid_search(
    query: str,
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    """Two-stage retrieval: TF-IDF recall + BM25 recall → RRF fusion.

    Combines the strengths of both ranking algorithms for better results.
    """
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []
    for slug in list_saved_topics():
        papers = load_papers(slug)
        for chunk in build_corpus_chunks(papers):
            texts.append(chunk["text"])
            metas.append({
                "paper_id": chunk["paper_id"],
                "title": chunk["title"],
                "chunk_idx": chunk["chunk_idx"],
                "topic": slug,
            })
    if not texts:
        return []

    # Stage 1: TF-IDF
    tfidf_idx = TFIDFIndex()
    tfidf_idx.add_documents(texts, metas)
    tfidf_results = tfidf_idx.query(query, top_k=top_k * 2)

    # Stage 2: BM25
    bm25_idx = BM25Index()
    bm25_idx.add_documents(texts, metas)
    bm25_results = bm25_idx.query(query, top_k=top_k * 2)

    # Fuse
    fused = reciprocal_rank_fusion(tfidf_results, bm25_results)
    return fused[:top_k]


# ===========================================================================
# TextRank Extractive Summarizer (graph-based, PageRank on sentences)
# ===========================================================================

def _sentence_similarity(s1: str, s2: str) -> float:
    """Jaccard-like overlap between two tokenised sentences."""
    t1 = set(_tokenize(s1))
    t2 = set(_tokenize(s2))
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / (math.log(len(t1) + 1) + math.log(len(t2) + 1))


def textrank_summarize(
    papers: Dict[str, Dict[str, Any]],
    num_sentences: int = 8,
    damping: float = 0.85,
    iterations: int = 30,
) -> str:
    """Multi-document extractive summarisation using TextRank.

    Builds a sentence-similarity graph, runs PageRank-style iterations,
    and returns the top-ranked sentences as a coherent summary.
    """
    # Collect all sentences with source info
    all_sents: List[Dict[str, str]] = []
    for pid, p in papers.items():
        for sent in re.split(r"(?<=[.!?])\s+", p["summary"].strip()):
            sent = sent.strip()
            if 30 < len(sent) < 500:
                all_sents.append({"text": sent, "title": p["title"], "paper_id": pid})

    n = len(all_sents)
    if n == 0:
        return "No suitable sentences found for summarisation."
    if n <= num_sentences:
        return " ".join(s["text"] for s in all_sents)

    # Build adjacency matrix (similarity graph)
    adj: List[List[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            sim = _sentence_similarity(all_sents[i]["text"], all_sents[j]["text"])
            adj[i][j] = sim
            adj[j][i] = sim

    # PageRank iteration
    scores = [1.0 / n] * n
    for _ in range(iterations):
        new_scores = [0.0] * n
        for i in range(n):
            total_weight = sum(adj[i])
            if total_weight == 0:
                new_scores[i] = (1 - damping) / n
                continue
            rank_sum = sum(
                adj[j][i] / max(sum(adj[j]), 1e-10) * scores[j]
                for j in range(n)
            )
            new_scores[i] = (1 - damping) / n + damping * rank_sum
        scores = new_scores

    # Select top sentences (maintain original order for readability)
    ranked = sorted(range(n), key=lambda i: -scores[i])
    selected_indices = sorted(ranked[:num_sentences])

    md = ["## Multi-Document Summary (TextRank)\n"]
    for idx in selected_indices:
        s = all_sents[idx]
        md.append(f"- {s['text']}  *— {s['title']}*")
    return "\n".join(md)


# ===========================================================================
# Named Entity Extraction (regex-based NER for academic text)
# ===========================================================================

_MODEL_PATTERNS = [
    r"\b(GPT-?[234]?o?|ChatGPT|BERT|RoBERTa|T5|LLaMA(?:-\d)?|Llama[\s-]?\d?"
    r"|PaLM(?:-?\d)?|Claude|Gemini|DALL[\s·-]?E[\s-]?\d?|Stable[\s-]?Diffusion"
    r"|ResNet[\s-]?\d*|VGG[\s-]?\d*|ViT|CLIP|Whisper|Codex"
    r"|Mistral|Falcon|Phi[\s-]?\d|Gemma|DeepSeek|Qwen"
    r"|LSTM|GRU|VAE|GAN|U-?Net|YOLO(?:v\d)?|EfficientNet"
    r"|XGBoost|LightGBM|Random[\s]Forest|SVM"
    r"|Transformer|Attention[\s]?Mechanism)\b",
]

_DATASET_PATTERNS = [
    r"\b(ImageNet|CIFAR[\s-]?\d*|MNIST|COCO|SQuAD|GLUE|SuperGLUE"
    r"|WikiText[\s-]?\d*|Common[\s]?Crawl|OpenWebText|The[\s]?Pile"
    r"|WMT[\s-]?\d*|SNLI|MultiNLI|SST[\s-]?\d?"
    r"|MS[\s-]?MARCO|Natural[\s]?Questions|HotpotQA"
    r"|AG[\s]?News|IMDb|Yelp|Amazon[\s]?Reviews"
    r"|PubMed|MIMIC[\s-]?III|ChestX[\s-]?ray"
    r"|LibriSpeech|VoxCeleb|AudioSet)\b",
]

_METRIC_PATTERNS = [
    r"\b(accuracy|precision|recall|F1[\s-]?score|BLEU[\s-]?\d*"
    r"|ROUGE[\s-]?[L12N]*|perplexity|AUC[\s-]?ROC|mAP|IoU"
    r"|FID|SSIM|PSNR|WER|CER|MRR|NDCG"
    r"|top[\s-]?[15k][\s]?accuracy|exact[\s]?match"
    r"|human[\s]?eval|pass@k)\b",
]

_TASK_PATTERNS = [
    r"\b(text[\s]?classification|sentiment[\s]?analysis|named[\s]?entity[\s]?recognition"
    r"|question[\s]?answering|machine[\s]?translation|summarization|summarisation"
    r"|object[\s]?detection|image[\s]?classification|semantic[\s]?segmentation"
    r"|speech[\s]?recognition|text[\s]?generation|code[\s]?generation"
    r"|dialogue[\s]?system|recommendation[\s]?system|anomaly[\s]?detection"
    r"|relation[\s]?extraction|information[\s]?retrieval"
    r"|reinforcement[\s]?learning|transfer[\s]?learning"
    r"|few[\s-]?shot[\s]?learning|zero[\s-]?shot|fine[\s-]?tuning)\b",
]


def extract_entities(
    papers: Dict[str, Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Extract named entities (models, datasets, metrics, tasks) from abstracts.

    Returns a dict with keys ``models``, ``datasets``, ``metrics``, ``tasks``.
    Each value is a list of ``{"name", "count", "papers": [paper_ids]}``.
    """
    categories = {
        "models": _MODEL_PATTERNS,
        "datasets": _DATASET_PATTERNS,
        "metrics": _METRIC_PATTERNS,
        "tasks": _TASK_PATTERNS,
    }

    results: Dict[str, Dict[str, Dict[str, Any]]] = {
        k: {} for k in categories
    }

    for pid, p in papers.items():
        blob = p["title"] + " " + p["summary"]
        for cat, patterns in categories.items():
            for pattern in patterns:
                for match in re.finditer(pattern, blob, re.IGNORECASE):
                    name = match.group(1).strip()
                    canonical = name.lower().replace("-", "").replace(" ", "")
                    if canonical not in results[cat]:
                        results[cat][canonical] = {
                            "name": name,
                            "count": 0,
                            "papers": [],
                        }
                    results[cat][canonical]["count"] += 1
                    if pid not in results[cat][canonical]["papers"]:
                        results[cat][canonical]["papers"].append(pid)

    # Convert to sorted lists
    output: Dict[str, List[Dict[str, Any]]] = {}
    for cat, entities in results.items():
        sorted_ents = sorted(entities.values(), key=lambda x: -x["count"])
        output[cat] = sorted_ents
    return output


# ===========================================================================
# Paper Clustering (Pure-Python K-Means on TF-IDF Vectors)
# ===========================================================================

import random

def _sparse_cosine(a: Dict[int, float], b: Dict[int, float]) -> float:
    """Cosine similarity between two sparse vectors."""
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in a)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _sparse_centroid(vectors: List[Dict[int, float]]) -> Dict[int, float]:
    """Compute mean centroid of sparse vectors."""
    if not vectors:
        return {}
    centroid: Dict[int, float] = {}
    for v in vectors:
        for k, val in v.items():
            centroid[k] = centroid.get(k, 0.0) + val
    n = len(vectors)
    return {k: v / n for k, v in centroid.items()}


def cluster_papers(
    papers: Dict[str, Dict[str, Any]],
    n_clusters: int = 3,
    max_iter: int = 20,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """K-Means clustering on TF-IDF paper vectors.

    Returns a list of cluster dicts::

        [{"cluster_id", "label", "papers": [{"paper_id", "title"}], "keywords"}]
    """
    pids = list(papers.keys())
    if len(pids) < n_clusters:
        n_clusters = max(1, len(pids))

    # Build TF-IDF vectors
    index = TFIDFIndex()
    texts = [papers[pid]["summary"] for pid in pids]
    index.add_documents(texts)
    vectors = index.tfidf_matrix  # sparse rows

    # K-Means++ init
    rng = random.Random(seed)
    centroids: List[Dict[int, float]] = [vectors[rng.randint(0, len(vectors) - 1)]]
    for _ in range(1, n_clusters):
        dists = []
        for v in vectors:
            min_d = min(1 - _sparse_cosine(v, c) for c in centroids)
            dists.append(min_d)
        total = sum(dists) or 1.0
        probs = [d / total for d in dists]
        r = rng.random()
        cumulative = 0.0
        chosen = 0
        for i, p in enumerate(probs):
            cumulative += p
            if cumulative >= r:
                chosen = i
                break
        centroids.append(vectors[chosen])

    # Iterate
    assignments = [0] * len(vectors)
    for _ in range(max_iter):
        # Assign
        new_assignments = []
        for v in vectors:
            sims = [_sparse_cosine(v, c) for c in centroids]
            new_assignments.append(sims.index(max(sims)))
        if new_assignments == assignments:
            break
        assignments = new_assignments
        # Update centroids
        for ci in range(n_clusters):
            cluster_vecs = [vectors[j] for j in range(len(vectors)) if assignments[j] == ci]
            if cluster_vecs:
                centroids[ci] = _sparse_centroid(cluster_vecs)

    # Build output
    inv_vocab = {idx: term for term, idx in index.vocab.items()}
    clusters: List[Dict[str, Any]] = []
    for ci in range(n_clusters):
        member_indices = [j for j in range(len(vectors)) if assignments[j] == ci]
        if not member_indices:
            continue
        # Cluster label from top centroid terms
        centroid = centroids[ci]
        top_terms = sorted(centroid.items(), key=lambda x: -x[1])[:5]
        label = ", ".join(inv_vocab.get(t[0], "?") for t in top_terms)
        cluster_papers_list = [
            {"paper_id": pids[j], "title": papers[pids[j]]["title"]}
            for j in member_indices
        ]
        clusters.append({
            "cluster_id": ci,
            "label": label,
            "papers": cluster_papers_list,
            "size": len(cluster_papers_list),
            "keywords": [inv_vocab.get(t[0], "?") for t in top_terms],
        })
    return clusters


# ===========================================================================
# Topic Drift / Evolution Analysis
# ===========================================================================

def analyze_topic_drift(
    papers: Dict[str, Dict[str, Any]],
    top_n: int = 8,
) -> Dict[str, Dict[str, Any]]:
    """Analyse how research focus shifts over time.

    Groups papers by year and extracts top keywords per period.
    Returns ``{year: {"keywords": [(word, count), ...], "methods": [...]}}``.
    """
    by_year = papers_by_year(papers)
    drift: Dict[str, Dict[str, Any]] = {}

    for year in sorted(by_year):
        year_papers = by_year[year]
        summaries = [p["summary"] for p in year_papers]
        kws = extract_keywords(summaries, top_n=top_n)
        methods_dict = extract_methodologies(
            {str(i): p for i, p in enumerate(year_papers)}
        )
        drift[year] = {
            "keywords": kws,
            "methods": list(methods_dict.items()),
            "paper_count": len(year_papers),
        }
    return drift


def format_topic_drift(
    papers: Dict[str, Dict[str, Any]],
) -> str:
    """Return a formatted Markdown report of topic drift over time."""
    drift = analyze_topic_drift(papers)
    if not drift:
        return "Not enough data for topic drift analysis."

    md = ["## Topic Evolution Over Time\n"]

    # Timeline with keywords
    for year, data in drift.items():
        md.append(f"### {year} ({data['paper_count']} papers)\n")
        md.append("**Top Keywords:**")
        for kw, cnt in data["keywords"][:6]:
            md.append(f"- {kw} ({cnt})")
        if data["methods"]:
            md.append("\n**Methods:**")
            for met, cnt in data["methods"][:3]:
                md.append(f"- {met} ({cnt})")
        md.append("")

    # Emerging / declining terms
    years = sorted(drift.keys())
    if len(years) >= 2:
        first_kws = {kw for kw, _ in drift[years[0]]["keywords"]}
        last_kws = {kw for kw, _ in drift[years[-1]]["keywords"]}
        emerging = last_kws - first_kws
        declining = first_kws - last_kws
        if emerging:
            md.append("### 📈 Emerging Terms")
            md.append(", ".join(f"**{t}**" for t in list(emerging)[:8]))
        if declining:
            md.append("\n### 📉 Declining Terms")
            md.append(", ".join(f"~~{t}~~" for t in list(declining)[:8]))

    return "\n".join(md)


# ===========================================================================
# Knowledge Graph Builder (concept co-occurrence network)
# ===========================================================================

def build_knowledge_graph(
    papers: Dict[str, Dict[str, Any]],
    top_concepts: int = 20,
) -> Dict[str, Any]:
    """Build a concept co-occurrence knowledge graph.

    Extracts key concepts (keywords) and finds which ones appear
    together in the same paper, creating edges with co-occurrence counts.

    Returns::

        {
            "nodes": [{"id": concept, "weight": count}],
            "edges": [{"source", "target", "weight"}],
            "papers_per_concept": {concept: [paper_ids]}
        }
    """
    # Extract concepts via keyword extraction per paper
    paper_concepts: Dict[str, List[str]] = {}
    global_counts: Counter = Counter()

    for pid, p in papers.items():
        kws = extract_keywords([p["summary"]], top_n=10, min_word_len=4)
        concepts = [kw for kw, _ in kws]
        paper_concepts[pid] = concepts
        global_counts.update(concepts)

    # Top N concepts form the graph nodes
    top = [c for c, _ in global_counts.most_common(top_concepts)]
    top_set = set(top)

    nodes = [{"id": c, "weight": global_counts[c]} for c in top]

    # Edges: co-occurrence in same paper
    edge_counts: Counter = Counter()
    for pid, concepts in paper_concepts.items():
        relevant = [c for c in concepts if c in top_set]
        for i in range(len(relevant)):
            for j in range(i + 1, len(relevant)):
                pair = tuple(sorted([relevant[i], relevant[j]]))
                edge_counts[pair] += 1

    edges = [
        {"source": pair[0], "target": pair[1], "weight": cnt}
        for pair, cnt in edge_counts.most_common(50)
        if cnt >= 1
    ]

    # Papers per concept
    ppc: Dict[str, List[str]] = {}
    for pid, concepts in paper_concepts.items():
        for c in concepts:
            if c in top_set:
                ppc.setdefault(c, []).append(pid)

    return {"nodes": nodes, "edges": edges, "papers_per_concept": ppc}


def format_knowledge_graph(
    papers: Dict[str, Dict[str, Any]],
) -> str:
    """Return a Markdown representation of the knowledge graph."""
    kg = build_knowledge_graph(papers)
    if not kg["nodes"]:
        return "Not enough data to build a knowledge graph."

    md = ["## Concept Knowledge Graph\n"]
    md.append(f"**{len(kg['nodes'])} concepts** connected by "
              f"**{len(kg['edges'])} co-occurrence edges**\n")

    md.append("### Key Concepts (nodes)\n")
    md.append("| Concept | Mentions | Connected Papers |")
    md.append("|---------|----------|------------------|")
    for node in kg["nodes"][:15]:
        n_papers = len(kg["papers_per_concept"].get(node["id"], []))
        md.append(f"| {node['id']} | {node['weight']} | {n_papers} |")

    md.append("\n### Strongest Connections (edges)\n")
    md.append("| Concept A | Concept B | Co-occurrences |")
    md.append("|-----------|-----------|----------------|")
    for edge in kg["edges"][:15]:
        md.append(f"| {edge['source']} | {edge['target']} | {edge['weight']} |")

    return "\n".join(md)
