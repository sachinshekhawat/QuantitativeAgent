# mcp_server.py
from mcp.server.fastmcp import FastMCP
import yfinance as yf
from langchain_community.tools import DuckDuckGoSearchRun
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Global variable to cache the vector database in memory
pdf_vector_store = None

mcp = FastMCP("FinanceLegalServer")

@mcp.tool()
def search_company_document(query: str, pdf_filename: str = "10k_report.pdf") -> str:
    """
    Searches a massive PDF document (like an SEC 10-K or legal filing) for specific compliance risk factors.
    Pass the query you want to search for.
    """
    global pdf_vector_store
    
    print(f"\n[MCP SERVER] Searching PDF for: '{query}'...\n")
    
    if not os.path.exists(pdf_filename):
        return f"Error: Could not find document {pdf_filename} in the root directory."

    # 1. Lazy-load and index the PDF only on the first tool call
    if pdf_vector_store is None:
        print(f"[MCP SERVER] First run detected. Indexing {pdf_filename}. This takes a few seconds...")
        try:
            # Load the 50+ page PDF
            loader = PyPDFLoader(pdf_filename)
            docs = loader.load()
            
            # Chunk the text so the LLM doesn't get overwhelmed
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            
            # Create a local vector database using free, local embeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            pdf_vector_store = FAISS.from_documents(splits, embeddings)
            print("[MCP SERVER] Indexing complete!")
            
        except Exception as e:
            return f"Failed to process PDF: {str(e)}"
            
    # 2. Perform the semantic search
    try:
        # Retrieve the top 3 most relevant chunks
        results = pdf_vector_store.similarity_search(query, k=3)
        
        # Format the retrieved text for the LLM
        formatted_results = "\n\n---\n\n".join([doc.page_content for doc in results])
        return f"Found the following relevant excerpts in {pdf_filename}:\n\n{formatted_results}"
    except Exception as e:
        return f"Search failed: {str(e)}"

    
@mcp.tool()
def get_market_data(ticker: str) -> str:
    """Fetches the current price and 5-day historical closing prices for a given ticker."""
    print(f"\n[MCP SERVER] Fetching market data for {ticker}...\n")
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        if hist.empty:
            return f"Could not find data for ticker: {ticker}"
        
        current_price = hist['Close'].iloc[-1]
        historical_str = hist['Close'].to_string()
        return f"Current Price: ${current_price:.2f}\nRecent 5-day history:\n{historical_str}"
    except Exception as e:
        return f"Error fetching data: {str(e)}"

@mcp.tool()
def calculate_moving_average(ticker: str, window: int = 20) -> str:
    """Calculates the Simple Moving Average (SMA) over a given number of days (default: 20) and compares it to current price."""
    print(f"\n[MCP SERVER] Calculating {window}-day SMA for {ticker}...\n")
    try:
        stock = yf.Ticker(ticker)
        # Fetch enough historical days to calculate rolling average
        hist = stock.history(period="60d")
        if hist.empty or len(hist) < window:
            return f"Not enough historical data to calculate {window}-day SMA for {ticker}."
        
        sma = hist['Close'].rolling(window=window).mean().iloc[-1]
        current_price = hist['Close'].iloc[-1]
        diff_pct = ((current_price - sma) / sma) * 100
        
        position = "ABOVE" if current_price >= sma else "BELOW"
        return (
            f"Ticker: {ticker}\n"
            f"Current Price: ${current_price:.2f}\n"
            f"{window}-Day SMA: ${sma:.2f}\n"
            f"Asset is trading {abs(diff_pct):.2f}% {position} its {window}-day moving average."
        )
    except Exception as e:
        return f"Error calculating SMA: {str(e)}"

@mcp.tool()
def search_regulatory_news(query: str) -> str:
    """Searches the web for recent regulatory news, SEC filings, or lawsuits regarding a company."""
    print(f"\n[MCP SERVER] Executing search for: {query}...\n")
    search = DuckDuckGoSearchRun()
    try:
        return search.invoke(query)
    except Exception as e:
        return f"Search failed: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")