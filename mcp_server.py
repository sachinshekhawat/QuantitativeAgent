# mcp_server.py
from mcp.server.fastmcp import FastMCP
import yfinance as yf
from langchain_community.tools import DuckDuckGoSearchRun

mcp = FastMCP("FinanceLegalServer")

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