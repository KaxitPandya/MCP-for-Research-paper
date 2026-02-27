# 🔬 arXiv MCP Research Assistant

> **AI-powered academic research platform** with **RAG retrieval**, **BM25 + TF-IDF hybrid search**, **TextRank summarization**, **K-Means paper clustering**, **knowledge graph**, **NER**, and an **autonomous multi-step agent** — built on the **Model Context Protocol (MCP)**.

[![Live App](https://img.shields.io/badge/Streamlit-Live_App-FF4B4B?logo=streamlit)](https://mcp-for-research-paper.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![FastMCP](https://img.shields.io/badge/FastMCP-MCP_Server-764ba2)](https://github.com/jlowin/fastmcp)
[![arXiv API](https://img.shields.io/badge/Data-arXiv_API-b31b1b?logo=arxiv)](https://arxiv.org/)

---

## ✨ Features

### 🧠 AI / RAG Intelligence

| Feature | Description |
|---------|-------------|
| **TF-IDF Semantic Search** | Pure-Python TF-IDF vectoriser with cosine similarity — no API keys, no sklearn |
| **BM25 Retriever** | Okapi BM25 ranking (the algorithm behind Elasticsearch/Lucene) |
| **Hybrid Search + RRF** | Two-stage TF-IDF + BM25 → Reciprocal Rank Fusion — production-grade retrieval |
| **Sentence-Level Chunking** | Abstracts split into ~300-char sentence chunks for fine-grained retrieval |
| **RAG Context Builder** | Retrieve top-k chunks and assemble grounded context for LLM prompts |
| **TextRank Summarizer** | Graph-based multi-document extractive summarization (PageRank on sentences) |
| **Named Entity Recognition** | Regex NER for AI models, datasets, metrics, and tasks (100+ patterns) |
| **Paper Clustering (K-Means)** | Unsupervised grouping on TF-IDF vectors with K-Means++ init and auto-labeling |
| **Paper Similarity Matrix** | Pairwise cosine similarity with interactive Plotly heatmap |
| **Query Expansion** | 25+ domain-specific synonym mappings for better recall |
| **9-Dimension Gap Analysis** | Scans abstracts against Scalability, Privacy, Explainability, Fairness, Reproducibility, Efficiency, Multi-Modal, Real-World Deployment, Human Evaluation |
| **Research Question Generator** | Combines gaps + keywords + methodologies into actionable questions |
| **Topic Drift / Evolution** | Track keyword emergence and decline year-over-year |
| **Knowledge Graph** | Concept co-occurrence network with interactive Plotly visualization |
| **Autonomous Research Agent** | 8-step pipeline: expand → search → chunk → index → retrieve → gaps → questions → review |

### 📊 Analysis & Visualisation

| Feature | Description |
|---------|-------------|
| **Literature Review** | Auto-generated comprehensive reviews with contribution analysis |
| **Trend Analysis** | Publication timeline, growth rate, emerging keywords |
| **Author Network** | Prolific-author ranking and collaboration-size distribution |
| **Interactive Charts** | Plotly: timeline, categories, keywords, methods, authors, similarity heatmap, knowledge graph |
| **Multi-Format Export** | BibTeX, APA citations, full Markdown review |

### ⚡ MCP Server (Model Context Protocol)

19 MCP tools exposed for **Claude Desktop**, **Cursor**, or any MCP client — including all AI/RAG tools.

---

## 🏗️ Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                      Streamlit Web UI                             │
│  Search · AI/RAG · Visualisations · Export · Chat                 │
└────────────────────────┬──────────────────────────────────────────┘
                         │
                  ┌──────▼──────┐
                  │  utils.py   │  Shared Core (~1600 lines)
                  │             │  ├─ arXiv API client (async)
                  │             │  ├─ TFIDFIndex + BM25Index classes
                  │             │  ├─ Hybrid retrieval + RRF
                  │             │  ├─ Sentence chunker
                  │             │  ├─ TextRank summarizer (PageRank)
                  │             │  ├─ Named entity extraction (NER)
                  │             │  ├─ K-Means paper clustering
                  │             │  ├─ Query expansion (synonym map)
                  │             │  ├─ Gap analysis engine (9 dims)
                  │             │  ├─ Topic drift / evolution
                  │             │  ├─ Knowledge graph builder
                  │             │  ├─ run_research_agent() — 8-step
                  │             │  ├─ NLP keyword extraction
                  │             │  └─ BibTeX/APA/MD export
                  └──────┬──────┘
                         │
           ┌─────────────┴──────────────┐
           │                            │
    ┌──────▼───────┐             ┌──────▼──────┐
    │  FastMCP     │             │  arXiv API  │
    │  Server      │             │  (HTTP/XML) │
    │  (19 tools)  │             └─────────────┘
    └──────────────┘
           ↕
    Claude Desktop / Cursor / any MCP client
```

### Retrieval Pipeline

```
User Query
    │
    ├──→ TF-IDF Ranking (cosine similarity)
    │                                        ──→ Reciprocal Rank Fusion ──→ Results
    └──→ BM25 Ranking (Okapi BM25 k1/b)

Alternative paths:
    Query → Expansion (25+ synonyms) → arXiv Search
    Query → Sentence Chunking → TF-IDF Index → Cosine Retrieval → RAG Context
```

### Analysis Pipeline

```
Papers ──→ TextRank (PageRank on sentence graph) ──→ Extractive Summary
Papers ──→ TF-IDF Vectors ──→ K-Means++ ──→ Auto-Labeled Clusters
Papers ──→ Regex NER ──→ Models / Datasets / Metrics / Tasks
Papers ──→ Keyword Extraction per Year ──→ Emerging / Declining Terms
Papers ──→ Concept Extraction ──→ Co-Occurrence Graph ──→ Knowledge Graph
Papers ──→ 9-Dimension Regex Scan ──→ Gap Analysis ──→ Research Questions
```

---

## 📂 Project Structure

| File | Lines | Purpose |
|------|-------|---------|
| `utils.py` | ~1600 | TF-IDF + BM25 engines, hybrid retrieval + RRF, TextRank summarizer, NER, K-Means clustering, topic drift, knowledge graph, gap analysis, research agent, arXiv API, export |
| `research_server.py` | ~600 | FastMCP server — 19 tool endpoints |
| `streamlit_app.py` | ~830 | Tabbed web UI with full AI/RAG dashboard |

---

## 🚀 Quick Start

### Live Demo

👉 **[mcp-for-research-paper.streamlit.app](https://mcp-for-research-paper.streamlit.app/)**

### Local Setup

```bash
git clone https://github.com/<your-username>/MCP-for-Research-paper.git
cd MCP-for-Research-paper
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### MCP Server (for Claude Desktop / Cursor)

```bash
python research_server.py
```

#### Claude Desktop config

```json
{
  "mcpServers": {
    "arxiv-research": {
      "command": "python",
      "args": ["path/to/research_server.py"]
    }
  }
}
```

---

## 🛠️ MCP Tools Reference

### Core Tools

| Tool | Description |
|------|-------------|
| `search_papers` | Search arXiv with query expansion |
| `analyze_papers` | Full literature review |
| `compare_papers` | Side-by-side paper comparison |
| `track_research_trends` | Trend analysis report |
| `export_bibtex` | BibTeX export |
| `export_citations` | APA or Markdown citations |

### AI / RAG Tools

| Tool | Description |
|------|-------------|
| `semantic_search` | TF-IDF search across all saved papers |
| `smart_search` | **Hybrid TF-IDF + BM25 with RRF** |
| `ask_papers` | RAG-style QA against paper corpus |
| `find_related` | Cosine similarity to find related papers |
| `summarize_corpus` | **TextRank multi-document summarization** |
| `extract_named_entities` | **NER: models, datasets, metrics, tasks** |
| `cluster_topic` | **K-Means paper clustering** |
| `topic_evolution` | **Temporal keyword drift analysis** |
| `knowledge_graph` | **Concept co-occurrence network** |
| `identify_gaps` | 9-dimension research gap analysis |
| `suggest_questions` | Auto-generated research questions |
| `research_agent` | Full 8-step autonomous pipeline |

---

## 🧪 Technical Highlights (Resume-Worthy)

- **Okapi BM25**: Full implementation with k1/b parameters, IDF scoring — the same algorithm powering Elasticsearch
- **Reciprocal Rank Fusion**: Merges TF-IDF and BM25 ranked lists (Cormack et al. 2009) — used by production search systems like Pinecone and Weaviate
- **TextRank / PageRank**: Graph-based sentence ranking for extractive multi-document summarization — the same algorithm Google was built on
- **K-Means++ Clustering**: Smart centroid initialization on TF-IDF vectors with automatic cluster labeling from top centroid terms
- **Knowledge Graph**: Concept co-occurrence network with interactive Plotly force-directed visualization
- **Named Entity Recognition**: 100+ regex patterns covering modern AI models (GPT-4, LLaMA, BERT...), datasets (ImageNet, SQuAD...), metrics (BLEU, F1...), and tasks
- **Pure-Python ML**: All algorithms (TF-IDF, BM25, K-Means, TextRank, NER) implemented from scratch — zero sklearn/numpy dependency
- **RAG without LLM API**: Full retrieval-augmented generation architecture (chunk → index → retrieve → assemble) without paid API keys
- **Multi-Step Agent**: 8-step autonomous pipeline mirroring LangChain/AutoGPT agent patterns
- **Model Context Protocol**: 19 MCP tools exposable to Claude Desktop, Cursor, or any MCP client
- **Async Throughout**: `aiohttp` for non-blocking arXiv API calls; Streamlit bridge handles sync ↔ async

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Commit your changes (`git commit -m "feat: add awesome feature"`)
4. Push & open a Pull Request

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
