# Agentic Financial Due Diligence Pipeline

A production-grade, multi-agent AI system built to automate quantitative financial analysis, regulatory compliance checks, and portfolio allocation. This project leverages **LangGraph** for stateful agent orchestration, the **Model Context Protocol (MCP)** for secure, decoupled tool execution, and **Local RAG** for parsing massive financial documents.

## 🏗 Architecture Overview

This system moves beyond basic chatbots by implementing a cyclic, state-machine workflow with specialized AI agents and a Human-in-the-Loop (HITL) approval process.

*   **Orchestrator (`agent.py`):** Uses LangGraph to manage the state and route tasks between three distinct agents:
    1.  **Quant Analyst:** Fetches live market data and calculates technical indicators (e.g., 20-day SMA).
    2.  **Compliance Officer:** Synthesizes internal document risks (via RAG) and external web risks (via search) to generate a quantitative Risk Score (0-100%).
    3.  **Portfolio Manager (Supervisor):** Synthesizes the data into a final position sizing and allocation thesis.
*   **Tooling Server (`mcp_server.py`):** A standalone FastMCP server that exposes real-world data tools to the agents over standard inter-process communication (stdio).
*   **Local RAG Pipeline:** Dynamically ingests large PDF documents (like SEC 10-K filings), chunks the text, and builds an in-memory FAISS vector database using local HuggingFace embeddings for semantic search.
*   **Memory & Safety:** Utilizes LangGraph's `MemorySaver` to persist state and interrupt the execution graph for human approval before the Portfolio Manager finalizes the thesis.

## 🛠 Tech Stack

*   **Frameworks:** LangGraph, LangChain
*   **Protocols:** Model Context Protocol (FastMCP, `langchain-mcp-adapters`)
*   **Vector DB & RAG:** FAISS, HuggingFace (`all-MiniLM-L6-v2`), PyPDF
*   **Data Sources:** `yfinance` (Market Data), `duckduckgo-search` (Regulatory News)
*   **LLM Engine:** Google Gemini (`gemini-1.5-flash` / `gemini-2.5-flash`) via `langchain-google-genai`

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have Python 3.10+ installed. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install langgraph langchain-core langchain-google-genai mcp langchain-mcp-adapters yfinance duckduckgo-search==5.3.1 python-dotenv pypdf faiss-cpu sentence-transformers langchain-huggingface
```

### 3. Environment Setup
Create a `.env` file in the root directory and add your DeepSeek API key:
```text
LLM_API_KEY=your_api_key_here # Gemini Api key
```
### 4. Run the Pipeline
The system automatically boots the MCP server as a subprocess and initializes the LangGraph client.

```bash
python agent.py
```
