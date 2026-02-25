# Study Guide: Advanced RAG Techniques

This document serves as a reference for the advanced Retrieval-Augmented Generation (RAG) techniques implemented in the HDB RAG system.

## 1. Structure-Aware Semantic Chunking
**Problem**: Fixed-size chunking (e.g., every 1000 characters) often cuts sentences in half or separates related paragraphs, leading to "context fragmentation."

**Solution**: 
- **Structural Splitting**: The parser now respects HTML structure (`<h1>`, `<h2>`, `<p>`). Chunks are broken at logical boundaries.
- **Context Injection**: Each chunk is prepended with its section header (e.g., `[Enhanced CPF Housing Grant] ...`). This ensures that even if a snippet is small, the LLM knows exactly which policy it belongs to.

---

## 2. BM25 (Best Matching 25)
**Technique**: A ranking function used by search engines to estimate the relevance of documents to a given search query based on the **keywords** appearing in each document.

**Why use it?**:
- **Semantic search (Vector)** is great at finding "meaning" but bad at finding exact "technical terms" (e.g., "MOP", "BTO", "HFE letter").
- **BM25** excels at exact keyword matching. If a user asks for "MOP", BM25 will find the exact "Minimum Occupation Period" documents even if the embedding model thinks "MOP" is too generic.

---

## 3. Hybrid Search
**Technique**: Combining Vector Search (Semantic) and BM25 (Keyword) search.

**How it works (Reciprocal Rank Fusion - RRF)**:
1. Retrieve Top-K results from Vector search.
2. Retrieve Top-K results from BM25 search.
3. Combine both lists using **RRF**, which gives higher scores to documents that appear at the top of *both* lists.
4. This captures both the *intent* (semantic) and the *specificity* (keyword) of the query.

---

## 4. Cross-Encoder Reranking
**Technique**: Using a second, more powerful model to evaluate the relationship between the query and the retrieved documents.

**Stages**:
- **Stage 1 (Bi-Encoder)**: Fast. Compares pre-computed embeddings. This is what Initial Retrieval does.
- **Stage 2 (Cross-Encoder)**: Slower but more accurate. It processes the query and document *together* to determine relevance.
- **Benefit**: It acts as a "precision filter," ensuring that the top-3 results passed to the LLM are truly the most relevant from a candidate pool of 10-20.

---

## 5. Query Transformation
**Goal**: Improving the query before it ever hits the index.

### Multi-Query Expansion
Generates variations of the user's question (e.g., "How to apply for BTO?" → "BTO application process", "BTO eligibility criteria"). This increases the chance of a "hit" in the index.

### HyDE (Hypothetical Document Embeddings)
- The model generates a *hypothetical* ideal answer to the question.
- We use the embedding of this *hypothetical answer* to search.
- **Why?**: Often, a question's embedding is very different from an answer's embedding. By searching with a "fake answer," we find real documents that "look like" answers.

---

## Summary of Implementation
| Technique | Implementation File | Key Class/Function |
|-----------|----------------------|----------------------|
| Chunking | `html_parser.py` | `chunk_text_by_structure` |
| Hybrid Search | `retriever.py` | `QueryFusionRetriever` |
| Reranking | `retriever.py` | `SentenceTransformerRerank` |
| Query expansion | `model.py` | `GenerateSearchQueries` |
| HyDE | `model.py` | `GenerateHypotheticalAnswer` |
