# DSPyer-web

A collection of experiments and implementations using various Agentic AI frameworks, including LlamaIndex, LangGraph, and DSPy. This repository is designed to explore different approaches to building agentic applications and evaluating their performance.

## Prerequisites

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) (recommended for dependency management)

## Setup

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd DSPyer-web
    ```

2.  **Set up the environment:**

    This project uses `uv` for dependency management.

    ```bash
    # Install dependencies
    uv sync
    ```

    Alternatively, if you are using standard pip:

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    pip install .
    ```

3.  **Environment Variables:**

    Copy the `.env.example` file to `.env` and fill in your API keys.

    ```bash
    cp .env.example .env
    ```

    Open `.env` and add your keys (e.g., `OPENAI_API_KEY`).


