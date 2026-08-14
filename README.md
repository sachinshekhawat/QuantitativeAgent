# Agentic Financial Due Diligence Pipeline

A production-grade, multi-agent AI system built to automate quantitative financial analysis, regulatory compliance checks, and portfolio allocation. This project leverages **LangGraph** for stateful agent orchestration and the **Model Context Protocol (MCP)** for secure, decoupled tool execution.

## 🏗 Architecture Overview

This system moves beyond basic chatbots by implementing a cyclic, state-machine workflow with specialized AI agents and a Human-in-the-Loop (HITL) approval process.

*   **Orchestrator (`agent.py`):** Uses LangGraph to manage the state and route tasks between three distinct agents:
    1.  **Quant Analyst:** Fetches live market data and calculates technical indicators (e.g., 20-day SMA).
    2.  **Compliance Officer:** Scours the web for regulatory news, SEC filings, and lawsuits to generate a quantitative Risk Score (0-100%).
    3.  **Portfolio Manager (Supervisor):** Synthesizes the data into a final position sizing and allocation thesis.
*   **Tooling Server (`mcp_server.py`):** A standalone FastMCP server that exposes real-world data tools to the agents over standard inter-process communication (stdio).
*   **Memory & Safety:** Utilizes LangGraph's `MemorySaver` to persist state and interrupt the execution graph for human approval before the Portfolio Manager finalizes the thesis.
*   **LLM Engine:** Powered by DeepSeek via the OpenAI API adapter for high-performance agentic reasoning and strict tool-calling.

## 🛠 Tech Stack

*   **Frameworks:** LangGraph, LangChain
*   **Protocols:** Model Context Protocol (FastMCP, `langchain-mcp-adapters`)
*   **Data Sources:** `yfinance` (Market Data), `duckduckgo-search` (Regulatory News)
*   **LLM:** Gemini (`gemini-chat-flash-2.5`)

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have Python 3.10+ installed. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

### Step 2: Install Dependencies
```bash
pip install langgraph langchain-core langchain-openai mcp langchain-mcp-adapters yfinance duckduckgo-search==5.3.1 python-dotenv

### 3. Environment Setup
Create a `.env` file in the root directory and add your DeepSeek API key:
```text
LLM_API_KEY=your_api_key_here # Gemini Api key

### 4. Run the Pipeline
The system automatically boots the MCP server as a subprocess and initializes the LangGraph client.

```bash
python agent.py
