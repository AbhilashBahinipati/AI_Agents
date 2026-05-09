# Pattern 02: Parallel Fan-out / Fan-in
# Domain: Wealth Management - Portfolio Risk Analysis
#
# An HNI client's portfolio requires simultaneous multi-dimensional analysis.
# Three specialist agents run in parallel, then results are synthesized
# into a unified risk report for the Relationship Manager.
#
# Use this when: Independent subtasks can run concurrently to reduce latency.

from typing import TypedDict
from langgraph.graph import StateGraph, END, START
from langgraph.types import Send
from config import get_llm

llm = get_llm()


# -- State ------------------------------------------------------------------

class PortfolioState(TypedDict):
    portfolio: str
    market_risk_analysis: str
    liquidity_risk_analysis: str
    concentration_risk_analysis: str
    final_report: str


# -- Parallel Specialist Nodes ----------------------------------------------

def analyze_market_risk(state: PortfolioState) -> dict:
    """Specialist 1: Assess market and volatility risk."""
    print("\nMarket Risk Analysis - Running...")
    prompt = f"""
    You are a market risk analyst at a wealth management firm.
    Analyze the market risk of this portfolio:
    - Beta relative to Nifty 50
    - Estimated Value at Risk (VaR) at 95% confidence
    - Sector-level drawdown exposure
    - Overall market risk rating: Low / Medium / High

    Portfolio: {state['portfolio']}
    """
    response = ""
    for chunk in llm.stream(prompt):
        print(chunk.content, end="", flush=True)
        response += chunk.content
    print()
    return {"market_risk_analysis": response}


def analyze_liquidity_risk(state: PortfolioState) -> dict:
    """Specialist 2: Assess how quickly positions can be exited."""
    print("\nLiquidity Risk Analysis - Running...")
    prompt = f"""
    You are a liquidity risk analyst at a wealth management firm.
    Analyze the liquidity risk of this portfolio:
    - Which holdings are illiquid (small-cap, locked-in instruments)
    - Percentage that can be liquidated within 3 trading days
    - Liquidity risk rating: Low / Medium / High

    Portfolio: {state['portfolio']}
    """
    response = ""
    for chunk in llm.stream(prompt):
        print(chunk.content, end="", flush=True)
        response += chunk.content
    print()
    return {"liquidity_risk_analysis": response}


def analyze_concentration_risk(state: PortfolioState) -> dict:
    """Specialist 3: Check for over-concentration in sectors or stocks."""
    print("\nConcentration Risk Analysis - Running...")
    prompt = f"""
    You are a portfolio construction analyst at a wealth management firm.
    Analyze the concentration risk of this portfolio:
    - Top 3 holdings as percentage of total portfolio
    - Any sector exceeding 30% allocation
    - Geographic concentration
    - Concentration risk rating: Low / Medium / High
    - Diversification recommendations

    Portfolio: {state['portfolio']}
    """
    response = ""
    for chunk in llm.stream(prompt):
        print(chunk.content, end="", flush=True)
        response += chunk.content
    print()
    return {"concentration_risk_analysis": response}


def synthesize_risk_report(state: PortfolioState) -> dict:
    """Fan-in: Combine all three analyses into a unified report for the RM."""
    print("\nSynthesizing Final Risk Report...")
    prompt = f"""
    You are a senior risk officer at a wealth management firm.
    Synthesize the three risk analyses into a single Risk Summary Report
    for the Relationship Manager.

    Format:
    - Executive Summary (2 sentences)
    - Risk Dashboard: Market Risk | Liquidity Risk | Concentration Risk (ratings)
    - Top 3 Action Items for the Relationship Manager
    - Overall Portfolio Risk Score: 1 to 10

    Market Risk Analysis:
    {state['market_risk_analysis']}

    Liquidity Risk Analysis:
    {state['liquidity_risk_analysis']}

    Concentration Risk Analysis:
    {state['concentration_risk_analysis']}
    """
    response = ""
    for chunk in llm.stream(prompt):
        print(chunk.content, end="", flush=True)
        response += chunk.content
    print()
    return {"final_report": response}


# -- Graph ------------------------------------------------------------------

def build_graph():
    graph = StateGraph(PortfolioState)

    graph.add_node("market_risk", analyze_market_risk)
    graph.add_node("liquidity_risk", analyze_liquidity_risk)
    graph.add_node("concentration_risk", analyze_concentration_risk)
    graph.add_node("synthesize", synthesize_risk_report)

    def fan_out(state: PortfolioState):
        return [
            Send("market_risk", state),
            Send("liquidity_risk", state),
            Send("concentration_risk", state),
        ]

    graph.add_conditional_edges(START, fan_out, ["market_risk", "liquidity_risk", "concentration_risk"])
    graph.add_edge("market_risk", "synthesize")
    graph.add_edge("liquidity_risk", "synthesize")
    graph.add_edge("concentration_risk", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


# -- Run --------------------------------------------------------------------

if __name__ == "__main__":
    app = build_graph()

    sample_portfolio = """
    Client: HNI Client, Rs. 2 Crore AUM
    Holdings:
    - Reliance Industries: 25% (Large Cap, Energy and Retail)
    - HDFC Bank: 20% (Large Cap, Banking)
    - Tata Motors: 15% (Mid Cap, Automobile)
    - Zomato: 10% (Small Cap, Consumer Tech)
    - Gold ETF (GOLDBEES): 10%
    - Infosys: 10% (Large Cap, IT)
    - SBI Small Cap Fund: 10% (Mutual Fund, 3-year lock-in)
    """

    print("Parallel Fan-out - Portfolio Risk Analysis")
    print("=" * 60)

    result = app.invoke({"portfolio": sample_portfolio})

    print("\n" + "=" * 60)
    print("FINAL RISK REPORT")
    print(result["final_report"])