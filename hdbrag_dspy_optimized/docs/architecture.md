# HDB RAG System Architecture (Grounded View)

This document provides a grounded overview of the HDB RAG system, mapping logical components to their physical implementation in the codebase.

## 1. Project Structure

The project is organized into a modular structure where data, logic, and evaluation are clearly separated.

```text
.
├── app.py                  # Main Chatbot UI (Gradio/Streamlit)
├── rag_optimizer.py        # DSPy optimization script (Program Compilation)
├── compare_retrieval.py    # Side-by-side comparison of Naive vs Improved retrieval
├── data/
│   ├── chunks.json         # Processed semantic chunks (Output of ingestion)
│   ├── index_storage/      # Persistent LlamaIndex vector store
│   ├── hdb_raw/            # Source HTML documentation
│   └── qa_split.json       # Train/Dev/Test dataset for DSPy optimization
├── docs/                   # Study guides and architectural documentation
└── src/
    ├── ingestion/
    │   └── html_parser.py  # Structure-aware chunking logic
    ├── signatures.py       # DSPy Signatures (QueryExpansion, HyDE, Answer)
    ├── retriever.py        # Hybrid Search + BM25 + Reranking logic
    └── model.py            # Core RAG Module (HDBRAG) and pipeline orchestration
```

---

## 2. Integrated System Flow (Vertical View)

The diagram below showing the vertical progression from offline data preparation to the online runtime orchestration.

```mermaid
graph TD
    %% Offline Side
    subgraph Offline ["1. Context-Aware Ingestion (Offline)"]
        direction TB
        HDB[data/hdb_raw] --> PARSE[src/ingestion/html_parser.py]
        PARSE --> CHUNKS[data/chunks.json]
        CHUNKS --> INDEX[src/retriever.py: get_hdb_index]
        INDEX --> STORE[(data/index_storage)]
    end

    %% Online Side
    subgraph Online ["2. Runtime Orchestration (Online)"]
        direction TB
        STORE -.->|Load Index| RET
        UserQ[User Question] --> SIGS[src/signatures.py]
        SIGS -->|GenerateSearchQueries| MQ[Multi-Query Expansion]
        SIGS -->|GenerateHypotheticalAnswer| HYDE[HyDE Answer]
        
        MQ & HYDE --> RET[src/retriever.py: HDBRetriever]
        
        subgraph Retrieval_Engine ["Retrieval Engine"]
            direction TB
            RET --> VECTOR[LlamaIndex Vector Search]
            RET --> BM25[BM25 Keyword Search]
            VECTOR & BM25 --> FUSION[Query Fusion / RRF]
            FUSION --> RERANK[SentenceTransformer Reranker]
        end
        
        RERANK -->|Context| GEN[src/model.py: GenerateAnswer]
        GEN --> FinalA[Final Answer]
    end

    Offline --> Online
```

---

## 3. Detailed DSPy Module Flow (Logic-Level)

The following diagram details the sequence of execution within the `HDBRAG` module in `src/model.py`, showing how the various DSPy signatures interact to deliver a grounded answer.

```mermaid
sequenceDiagram
    participant User
    participant HDBRAG as HDBRAG (model.py)
    participant GSG as GenerateSearchQueries (Predict)
    participant GHA as GenerateHypotheticalAnswer (Predict)
    participant RET as HDBRetriever (retriever.py)
    participant GA as GenerateAnswer (ChainOfThought)

    User->>HDBRAG: question "What is MOP?"
    
    rect rgb(240, 240, 240)
    Note over HDBRAG, GSG: Stage 1: Query Expansion
    HDBRAG->>GSG: question
    GSG-->>HDBRAG: ["MOP requirements", "Minimum Occupation Period HDB"]
    end

    rect rgb(220, 230, 250)
    Note over HDBRAG, GHA: Stage 2: HyDE (Hypothetical Doc)
    HDBRAG->>GHA: question
    GHA-->>HDBRAG: "MOP is the time you must reside in your flat... (hypothetical)"
    end

    HDBRAG->>HDBRAG: Collect all queries: [original + expanded + hyde]

    rect rgb(230, 250, 230)
    Note over HDBRAG, RET: Stage 3: Multi-Pass Retrieval
    HDBRAG->>RET: list of queries
    RET->>RET: Hybrid Search (Vector + BM25) for EACH query
    RET->>RET: Deduplicate & Rerank via Cross-Encoder
    RET-->>HDBRAG: Top-K Context Snippets
    end

    rect rgb(250, 240, 230)
    Note over HDBRAG, GA: Stage 4: Grounded Generation
    HDBRAG->>GA: question + unique_context
    GA->>GA: Think (Chain of Thought)
    GA-->>HDBRAG: final_answer
    end

    HDBRAG-->>User: prediction (answer + used_context)
```

### Logical Breakdown of DSPy Components

| Component | Class | Signature | Responsibility |
|-----------|-------|-----------|----------------|
| **`generate_queries`** | `dspy.Predict` | `GenerateSearchQueries` | Breaks down complex questions into searchable terms to maximize "recall." |
| **`generate_hyde`** | `dspy.Predict` | `GenerateHypotheticalAnswer` | Bridges the "query-answer" gap by searching for documents similar to the *intended* answer. |
| **`retriever`** | `dspy.Retrieve` | N/A (Custom) | Operates as a black-box multi-query retrieval engine using LlamaIndex. |
| **`generate_answer`** | `dspy.ChainOfThought`| `GenerateAnswer` | Synthesizes the final response, ensuring every claim is backed by retrieved snippets. |

---

## 3. Core Component Implementation

### A. Context-Aware Ingestion (`src/ingestion/html_parser.py`)
- **Logic**: Uses BeautifulSoup to extract structure. Instead of splitting by character count, it splits at header tags (`h1-h6`) and paragraph boundaries.
- **Grounded Benefit**: Prevents the system from retrieving a "fragment" of a sentence. It injects the `[Section Header]` into the text itself, so the embedding model has more context to work with.

### B. Two-Stage Retrieval (`src/retriever.py`)
- **Hybrid Retrieval**: Combines `VectorStoreIndex` (semantic) with `BM25Retriever` (keyword).
- **Reranker**: Uses `SentenceTransformerRerank` with the `cross-encoder/ms-marco-TinyBERT-L-2-v2` model. This is the "brain" that filters out irrelevant matches from the hybrid pool.
- **Persistence**: Storage is handled via `StorageContext` and persisted to `data/index_storage`.

### C. DSPy Program Orchestration (`src/model.py` & `src/signatures.py`)
- **Model**: The `HDBRAG` module in `model.py` orchestrates the entire pipeline. It's a `dspy.Module` that can be "compiled" (optimized).
- **Transformation**: Before searching, `model.py` calls `generate_queries` and `generate_hyde` (defined in `signatures.py`) to expand the search scope.
- **Deduplication**: Since multiple queries are used, `model.py` implements a set-based deduplication to ensure the LLM doesn't see the same text twice.

### D. Optimization & Evaluation (`rag_optimizer.py`)
- **Process**: This script loads `data/qa_split.json` and uses the `MIPROv2` optimizer to tune the instructions for the signatures in `src/signatures.py`.
- **Result**: Generates `data/optimized_rag.json`, which contains the "compiled" prompts that perform significantly better than the hand-written versions.
