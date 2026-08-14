# agent.py
import os
import operator
import asyncio
import re
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient

# 1. Extend State Schema to include risk_score
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    ticker_symbol: str
    compliance_risk_detected: bool
    risk_score: int # Risk Score (0 - 100%)
    final_analysis: str

async def main():
    load_dotenv()
    os.environ["GOOGLE_API_KEY"] = os.getenv("LLM_API_KEY", "")
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

    client = MultiServerMCPClient({
        "finance_server": {
            "command": "python",
            "args": ["mcp_server.py"], # This automatically starts the server!
            "transport": "stdio",
        }
    })
    
    print("Connecting to MCP Server and discovering tools dynamically...")
    
    # NO MORE async with! Just fetch the tools directly.
    mcp_tools = await client.get_tools()
    llm_with_tools = llm.bind_tools(mcp_tools)
    
    # 2. Quant Analyst Node
    def quant_analyst_node(state: AgentState):
        messages = state.get("messages", [])
        ticker = state.get("ticker_symbol", "Unknown")
        prompt = SystemMessage(
            content=f"You are a Quant Analyst. Use your available tools to fetch market data AND "
                    f"calculate the 20-day moving average for {ticker}. Summarize price action vs technical level."
        )
        response = llm_with_tools.invoke([prompt] + list(messages))
        return {"messages": [response]}

    # 3. Compliance Node with Risk Percentage Scoring
    def compliance_node(state: AgentState):
        messages = state.get("messages", [])
        ticker = state.get("ticker_symbol", "Unknown")
        prompt = SystemMessage(
            content=f"You are a Compliance & Risk Officer. Search regulatory news for {ticker}. "
                    f"Evaluate legal risks, market volatility, and moving average stance. "
                    f"At the end of your analysis, specify a quantitative Risk Score on a new line strictly in this format: "
                    f"'RISK SCORE: X%' (where X is an integer from 0 to 100). "
                    f"If legal/regulatory issues are present, set X >= 60."
        )
        response = llm_with_tools.invoke([prompt] + list(messages))
        content_text = str(response.content)
        
        # Extract Risk Score using Regex
        score_match = re.search(r"RISK SCORE:\s*(\d+)%", content_text, re.IGNORECASE)
        extracted_score = int(score_match.group(1)) if score_match else 20
        risk_detected = extracted_score >= 50 or "RISK DETECTED" in content_text
        
        return {
            "messages": [response],
            "risk_score": extracted_score,
            "compliance_risk_detected": risk_detected
        }

    # 4. Supervisor Node
    def supervisor_node(state: AgentState):
        score = state.get("risk_score", 0)
        ticker = state.get("ticker_symbol", "Unknown")
        prompt = SystemMessage(
            content=f"You are the Lead Portfolio Manager. Review the quant and legal findings for {ticker}. "
                    f"The evaluated risk score is {score}%. "
                    f"Provide a 2-paragraph final decision on position sizing and portfolio allocation."
        )
        response = llm.invoke([prompt] + list(state["messages"]))
        return {"final_analysis": response.content}

    # 5. Graph Construction
    workflow = StateGraph(AgentState)
    workflow.add_node("quant_analyst", quant_analyst_node)
    workflow.add_node("compliance", compliance_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("tools", ToolNode(mcp_tools))

    workflow.add_edge(START, "quant_analyst")
    workflow.add_conditional_edges("quant_analyst", lambda s: "tools" if s["messages"][-1].tool_calls else "compliance")
    workflow.add_conditional_edges("compliance", lambda s: "tools" if s["messages"][-1].tool_calls else "supervisor")
    
    def route_from_tools(s):
        caller_message = s["messages"][-2]
        caller_content = caller_message.content
        
        # If the content is a string, make it lowercase
        if isinstance(caller_content, str):
            text_to_check = caller_content.lower()
        # If the content is a list, extract text from the dictionaries
        elif isinstance(caller_content, list):
            text_to_check = " ".join([
                item.get("text", "").lower() 
                for item in caller_content if isinstance(item, dict)
            ])
        else:
            text_to_check = ""
            
        # Also check the tool_calls themselves for clues if the content is empty
        if not text_to_check and hasattr(caller_message, "tool_calls"):
            for tc in caller_message.tool_calls:
                if tc["name"] in ["get_market_data", "calculate_moving_average"]:
                    return "quant_analyst"
                elif tc["name"] == "search_regulatory_news":
                    return "compliance"
            
        return "quant_analyst" if "quant" in text_to_check else "compliance"
        
    workflow.add_conditional_edges("tools", route_from_tools)
    workflow.add_edge("supervisor", END)

    app = workflow.compile(checkpointer=MemorySaver(), interrupt_before=["supervisor"])
    
    # 6. Execution Loop
    print("\nStarting execution...")
    config = {"configurable": {"thread_id": "mcp_risk_score_2"}}
    initial_state = {
        "messages": [HumanMessage(content="Analyze this asset for our portfolio.")], 
        "ticker_symbol": "VO",
        "compliance_risk_detected": False,
        "risk_score": 0,
        "final_analysis": ""
    }
    
    async for event in app.astream(initial_state, config=config):
        for k in event:
            print(f"Finished node: {k}")
            
    state_snapshot = app.get_state(config)
    if state_snapshot.next:
        current_state = state_snapshot.values
        score = current_state.get("risk_score", 0)
        
        print("\n" + "="*50)
        print(f"⏸️  HUMAN-IN-THE-LOOP INTERCEPT | Risk Score: {score}%")
        print("="*50)
        
        latest_msg = current_state["messages"][-1].content
        print("\nCompliance & Technical Report:")
        if isinstance(latest_msg, list):
            print(latest_msg[0].get("text", ""))
        else:
            print(latest_msg)
            
        user_input = input(f"\n👉 Risk score is {score}%. Approve proceeding to Supervisor? (approve/abort): ")
        
        if user_input.lower().strip() == 'approve':
            print("\nResuming execution into Supervisor Node...")
            async for event in app.astream(None, config=config):
                pass
        else:
            print("\n❌ Execution aborted by Human Supervisor.")
            
    print("\n=== FINAL PORTFOLIO MANAGER THESIS ===")
    print(app.get_state(config).values.get("final_analysis", "No thesis generated."))
    
    

if __name__ == "__main__":
    asyncio.run(main())