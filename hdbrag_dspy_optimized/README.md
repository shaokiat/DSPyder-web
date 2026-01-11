# DSPy HDB Optimized RAG

A RAG system designed to answer questions about HDB BTO grants and eligibility, leveraging **DSPy** for optimization and **Ollama** for local inference.

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.9+**
- **[uv](https://github.com/astral-sh/uv)** (Python package manager)
- **[Ollama](https://ollama.com/)** (for local LLMs)

### 2. Model Setup
Pull the required local model using Ollama:
```bash
# Default model for the chatbot and optimization
ollama pull qwen3:0.6b
```

### 3. Installation
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd dspy_hdb_optimized_rag
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Set up environment variables:
   Create a `.env` file in the root directory:
   ```bash
   OPENAI_API_KEY=your_api_key_here  # Required for data generation & OpenAI mode
   ```

---

## 📚 Data Pipeline

Before running the chatbot, you must ingest the data.

### Step 1: Download Data
Download raw HDB documentation.
```bash
uv run python src/ingestion/hdb_downloader.py
```

### Step 2: Parse & Chunk
Clean data and create chunks.
```bash
uv run python src/ingestion/html_parser.py
```
**Output**: `data/chunks.json`

### Step 3: Generate QA Pairs (Optional but Recommended)
Generate synthetic QA pairs for evaluation and optimization. Requires `OPENAI_API_KEY`.
```bash
uv run python src/ingestion/qa_generator.py
```
**Output**: `data/qa_pairs.json`

---

## 🤖 Running the Chatbot

Start the interactive terminal chatbot.

### Local Mode (Default)
Uses `qwen3:0.6b` running on Ollama.
```bash
uv run python app.py
```

### OpenAI Mode
Uses `gpt-4o-mini`. Requires `OPENAI_API_KEY`.
```bash
uv run python app.py --model openai
```

---

## ⚡ RAG Optimization

Improve the RAG system's accuracy using DSPy's `MIPROv2` optimizer. This process selects the best few-shot prompt examples.

**Note**: This script uses `qwen3:0.6b` by default for better evaluation capabilities.

```bash
uv run python rag_optimizer.py
```

### What does this script do?
1. **Load Data**: Splits `data/qa_pairs.json` into training and testing sets.
2. **Evaluate**: Benchmarks the initial (uncompiled) RAG pipeline.
3. **Optimize**: Runs MIPROv2 to find the optimal prompt instructions and examples.
4. **Save**: Exports the optimized program to `data/optimized_rag_qwen3:0.6b.json`.
5. **Verify**: Runs a final benchmark to show improvement.

---

## 📊 Efficiency Comparison

| System | smollm2:360m | qwen3:0.6b |
| :--- | :--- | :--- |
| **Unoptimized** | - | 31.0 / 40 (77.5%) |
| **Optimized (MIPROv2)** | - | 29.0 / 40 (72.5%) |

*Note: Metrics may vary based on the evaluation dataset size and randomness.*
