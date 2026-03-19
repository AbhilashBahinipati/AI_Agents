# Pattern 06: Human-in-the-Loop (HITL)
# Domain: Wealth Management - High-Value Trade Approval

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from config import get_llm

llm = get_llm()


# -- State ------------------------------------------------------------------

class TradeState(TypedDict):
    client_profile: str
    market_signal: str
    trade_recommendation: str
    risk_check: str
    human_decision: Optional[str]
    human_notes: Optional[str]
    execution_result: str


# -- Agent Nodes ------------------------------------------------------------

def generate_trade_recommendation(state: TradeState) -> TradeState:
    print("\n[AI Advisor] Generating trade recommendation...")
    prompt = f"""
    You are an AI investment advisor at a wealth management firm.
    Based on the client profile and market signal, recommend a specific trade.

    Include:
    - Trade action: BUY or SELL
    - Asset name and exchange
    - Quantity and indicative price
    - Rationale (2 to 3 sentences)
    - Risk level: Low / Medium / High
    - Trade value in rupees

    Client Profile: {state['client_profile']}
    Market Signal: {state['market_signal']}
    """
    response = ""
    for chunk in llm.stream(prompt):
        print(chunk.content, end="", flush=True)
        response += chunk.content
    print()
    return {**state, "trade_recommendation": response}


def run_risk_check(state: TradeState) -> TradeState:
    print("\n[Risk Engine] Validating trade against client guardrails...")
    prompt = f"""
    You are a risk compliance engine at a wealth management firm.
    Validate this trade recommendation against the following rules:
    - Single trade value must not exceed 10% of client AUM
    - Trade must not increase any single sector concentration beyond 35%
    - Leverage and derivatives are not permitted for this client category

    Output: PASS or FAIL with specific reason.

    Client Profile: {state['client_profile']}
    Trade Recommendation: {state['trade_recommendation']}
    """
    response = ""
    for chunk in llm.stream(prompt):
        print(chunk.content, end="", flush=True)
        response += chunk.content
    print()
    return {**state, "risk_check": response}


def request_human_approval(state: TradeState) -> TradeState:
    """HITL Checkpoint: Pause and present to the Relationship Manager."""
    print("\n" + "=" * 60)
    print("HUMAN APPROVAL REQUIRED")
    print("=" * 60)
    print(f"\nCLIENT PROFILE:\n{state['client_profile']}")
    print(f"\nMARKET SIGNAL:\n{state['market_signal']}")
    print(f"\nAI TRADE RECOMMENDATION:\n{state['trade_recommendation']}")
    print(f"\nRISK CHECK RESULT:\n{state['risk_check']}")
    print("\n" + "-" * 60)

    decision = input("\nRelationship Manager Decision - Enter APPROVE / REJECT / MODIFY:<details>: ").strip()
    notes = input("Notes (optional): ").strip()

    return {**state, "human_decision": decision, "human_notes": notes}


def route_after_human(state: TradeState) -> str:
    decision = (state.get("human_decision") or "").upper()
    if decision == "APPROVE":
        return "execute"
    elif decision == "REJECT":
        return "rejected"
    elif decision.startswith("MODIFY"):
        return "modify"
    return "rejected"


def modify_and_execute(state: TradeState) -> TradeState:
    print("\n[AI Advisor] Revising trade based on RM instructions...")
    modification = state["human_decision"].replace("MODIFY:", "").strip()
    prompt = f"""
    The Relationship Manager has requested a modification to the trade.
    Revise the trade recommendation accordingly and produce a final trade ticket.

    Original Recommendation: {state['trade_recommendation']}
    Modification Request: {modification}
    RM Notes: {state.get('human_notes', '')}
    """
    response = ""
    for chunk in llm.stream(prompt):
        print(chunk.content, end="", flush=True)
        response += chunk.content
    print()
    return {**state, "trade_recommendation": response, "human_decision": "APPROVE"}


def execute_trade(state: TradeState) -> TradeState:
    print("\n[Execution System] Processing trade...")
    prompt = f"""
    You are a trade execution system at a wealth management firm.
    Generate a trade confirmation for this approved trade.

    Include:
    - Trade confirmation number (format: TRD-2025-XXXXX)
    - Execution timestamp
    - Filled price (add realistic slippage of plus or minus 0.2%)
    - Brokerage and STT charges
    - Net consideration
    - Settlement date (T+1 for equities)

    Trade: {state['trade_recommendation']}
    """
    response = ""
    for chunk in llm.stream(prompt):
        print(chunk.content, end="", flush=True)
        response += chunk.content
    print()
    return {**state, "execution_result": f"TRADE EXECUTED SUCCESSFULLY\n\n{response}"}


def reject_trade(state: TradeState) -> TradeState:
    reason = state.get("human_notes") or "Rejected by Relationship Manager without specific reason."
    print(f"\nTrade rejected. Reason: {reason}")
    return {**state, "execution_result": f"TRADE REJECTED\n\nReason: {reason}"}


# -- Graph ------------------------------------------------------------------

def build_graph():
    memory = MemorySaver()
    graph = StateGraph(TradeState)

    graph.add_node("recommend", generate_trade_recommendation)
    graph.add_node("risk_check", run_risk_check)
    graph.add_node("human_approval", request_human_approval)
    graph.add_node("modify", modify_and_execute)
    graph.add_node("execute", execute_trade)
    graph.add_node("rejected", reject_trade)

    graph.set_entry_point("recommend")
    graph.add_edge("recommend", "risk_check")
    graph.add_edge("risk_check", "human_approval")
    graph.add_conditional_edges(
        "human_approval",
        route_after_human,
        {"execute": "execute", "rejected": "rejected", "modify": "modify"}
    )
    graph.add_edge("modify", "execute")
    graph.add_edge("execute", END)
    graph.add_edge("rejected", END)

    return graph.compile(checkpointer=memory)


# -- Run --------------------------------------------------------------------

if __name__ == "__main__":
    app = build_graph()
    config = {"configurable": {"thread_id": "trade-session-001"}}

    client_profile = """
    Client: Priya Nair, HNI
    AUM: Rs. 4.2 Crore
    Risk Profile: Moderate
    Current Allocation: 55% equities, 30% debt mutual funds, 15% gold
    Sector Exposure: IT 30%, Banking 25%, Pharma 10%, Others
    Investment Horizon: 7 years
    Restrictions: No derivatives, no intraday, no leverage
    """

    market_signal = """
    Signal: HDFC Bank (NSE: HDFCBANK) underperformed Nifty Bank by 12% over past 3 months.
    Trigger: Q3FY25 results beat on NII (up 10% YoY) and asset quality stable (GNPA 1.24%).
    Analyst Consensus: 9 out of 12 analysts have BUY with average target Rs. 1,950.
    Technical: Stock at 200-DMA support at Rs. 1,680. RSI at 38, oversold territory.
    CMP: Rs. 1,695
    """

    print("Human-in-the-Loop - Wealth Management Trade Approval")
    print("=" * 60)

    result = app.invoke({
        "client_profile": client_profile,
        "market_signal": market_signal,
        "human_decision": None,
        "human_notes": None,
    }, config=config)

    print("\n" + "=" * 60)
    print("FINAL OUTCOME")
    print(result["execution_result"])