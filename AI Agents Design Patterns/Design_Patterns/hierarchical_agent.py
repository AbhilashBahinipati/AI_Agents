# Pattern 05: Hierarchical (Supervisor / Master-Slave)
# Domain: Investment Banking - Merger and Acquisition Deal Coordination
#
# An M&A deal requires coordinated action across multiple workstreams.
# A Supervisor agent decomposes the deal brief and dispatches to
# specialist sub-agents: valuation, legal, synergy, and financing.
# The supervisor then synthesizes outputs into an IC (Investment Committee) memo.
#
# Use this when: Complex multi-domain tasks need a coordinating agent
# that dynamically delegates to specialists.

from typing import TypedDict
from langgraph.graph import StateGraph, END
from config import get_llm

llm = get_llm()


# -- State ------------------------------------------------------------------

class DealState(TypedDict):
    deal_brief: str
    coordination_plan: str
    valuation_analysis: str
    legal_assessment: str
    synergy_analysis: str
    financing_structure: str
    ic_memo: str


# -- Supervisor Node --------------------------------------------------------

def supervisor_plan(state: DealState) -> dict:
    """Supervisor: Assess the deal and define the workstream action plan."""
    print("\nSupervisor - Creating Coordination Plan...")
    prompt = f"""
    You are the Managing Director overseeing an M&A transaction.
    Review the deal brief and define the key priorities for each workstream:

    - VALUATION: Methodology to use, key metrics to anchor
    - LEGAL: Primary legal and regulatory concerns to assess
    - SYNERGIES: Revenue and cost synergy areas to quantify
    - FINANCING: Recommended deal financing structure and leverage considerations

    Be specific and action-oriented. This plan will guide the specialist teams.

    Deal Brief: {state['deal_brief']}
    """
    response = ""
    for chunk in llm.stream(prompt):
        print(chunk.content, end="", flush=True)
        response += chunk.content
    print()
    return {"coordination_plan": response}


# -- Sub-Agent Nodes --------------------------------------------------------

def valuation_agent(state: DealState) -> dict:
    print("\nValuation Workstream - Running...")
    prompt = f"""
    You are a valuation analyst in an M&A transaction.
    Perform a deal valuation covering:
    - DCF valuation with key assumptions (WACC, terminal growth rate)
    - Comparable company analysis (EV/EBITDA, P/E multiples)
    - Precedent transaction multiples
    - Implied equity value range and recommended offer price

    Coordination Plan: {state['coordination_plan']}
    Deal Brief: {state['deal_brief']}
    """
    response = ""
    for chunk in llm.stream(prompt):
        print(chunk.content, end="", flush=True)
        response += chunk.content
    print()
    return {"valuation_analysis": response}


def legal_agent(state: DealState) -> dict:
    print("\nLegal Workstream - Running...")
    prompt = f"""
    You are a legal and regulatory advisor on an M&A transaction.
    Assess the following:
    - CCI (Competition Commission of India) filing requirement
    - FEMA and RBI approvals if cross-border
    - Key legal risks and representations required
    - Recommended due diligence areas for legal team

    Coordination Plan: {state['coordination_plan']}
    Deal Brief: {state['deal_brief']}
    """
    response = ""
    for chunk in llm.stream(prompt):
        print(chunk.content, end="", flush=True)
        response += chunk.content
    print()
    return {"legal_assessment": response}


def synergy_agent(state: DealState) -> dict:
    print("\nSynergy Workstream - Running...")
    prompt = f"""
    You are a synergy analyst on an M&A transaction.
    Quantify potential synergies:
    - Revenue synergies: cross-sell opportunities, geographic expansion, pricing power
    - Cost synergies: headcount rationalization, procurement savings, facility consolidation
    - Timeline to achieve synergies (Year 1, Year 2, Year 3)
    - Total NPV of synergies

    Coordination Plan: {state['coordination_plan']}
    Deal Brief: {state['deal_brief']}
    """
    response = ""
    for chunk in llm.stream(prompt):
        print(chunk.content, end="", flush=True)
        response += chunk.content
    print()
    return {"synergy_analysis": response}


def financing_agent(state: DealState) -> dict:
    print("\nFinancing Workstream - Running...")
    prompt = f"""
    You are a leveraged finance specialist on an M&A transaction.
    Recommend a deal financing structure:
    - Equity vs debt split
    - Debt instruments: term loans, NCDs, mezzanine finance
    - Pro forma leverage ratios (Net Debt/EBITDA post-deal)
    - Interest coverage and debt serviceability assessment
    - Key financing risks

    Coordination Plan: {state['coordination_plan']}
    Deal Brief: {state['deal_brief']}
    """
    response = ""
    for chunk in llm.stream(prompt):
        print(chunk.content, end="", flush=True)
        response += chunk.content
    print()
    return {"financing_structure": response}


def supervisor_synthesize(state: DealState) -> dict:
    """Supervisor: Consolidate all workstream outputs into an IC memo."""
    print("\nSupervisor - Synthesizing Investment Committee Memo...")
    prompt = f"""
    You are the Managing Director preparing an Investment Committee memo
    for a proposed M&A transaction.

    Synthesize all workstream findings into a structured IC memo:

    1. Transaction Overview (2 sentences)
    2. Strategic Rationale
    3. Valuation Summary and Offer Price Recommendation
    4. Key Synergies (with NPV)
    5. Financing Structure
    6. Key Risks and Mitigants
    7. Regulatory Considerations
    8. Recommendation: PROCEED / PROCEED WITH CONDITIONS / DO NOT PROCEED

    VALUATION: {state['valuation_analysis']}
    LEGAL: {state['legal_assessment']}
    SYNERGIES: {state['synergy_analysis']}
    FINANCING: {state['financing_structure']}
    """
    response = ""
    for chunk in llm.stream(prompt):
        print(chunk.content, end="", flush=True)
        response += chunk.content
    print()
    return {"ic_memo": response}


# -- Graph ------------------------------------------------------------------

def build_graph():
    graph = StateGraph(DealState)

    graph.add_node("supervisor_plan", supervisor_plan)
    graph.add_node("valuation", valuation_agent)
    graph.add_node("legal", legal_agent)
    graph.add_node("synergy", synergy_agent)
    graph.add_node("financing", financing_agent)
    graph.add_node("supervisor_synthesize", supervisor_synthesize)

    graph.set_entry_point("supervisor_plan")
    graph.add_edge("supervisor_plan", "valuation")
    graph.add_edge("supervisor_plan", "legal")
    graph.add_edge("supervisor_plan", "synergy")
    graph.add_edge("supervisor_plan", "financing")
    graph.add_edge("valuation", "supervisor_synthesize")
    graph.add_edge("legal", "supervisor_synthesize")
    graph.add_edge("synergy", "supervisor_synthesize")
    graph.add_edge("financing", "supervisor_synthesize")
    graph.add_edge("supervisor_synthesize", END)

    return graph.compile()


# -- Run --------------------------------------------------------------------

if __name__ == "__main__":
    app = build_graph()

    deal_brief = """
    Transaction: Proposed acquisition of Medi-Assist Healthcare TPA Pvt. Ltd.
    Acquirer: A large listed private sector insurance company
    Target Business: India's largest Third Party Administrator (TPA) for health insurance.
      Processes over 2 crore claims annually. Revenues of Rs. 850 Cr (FY24), EBITDA Rs. 160 Cr.
    Deal Size: Acquirer is evaluating 100% buyout.
    Strategic Intent: Vertical integration to control claims processing, reduce fraud,
      improve turnaround times, and capture data for AI-led underwriting.
    Cross-border element: No. Both entities are Indian.
    Key Concern: CCI review likely given market share. Integration complexity.
    """

    print("Hierarchical Supervisor - M&A Deal Coordination")
    print("=" * 60)

    result = app.invoke({"deal_brief": deal_brief})

    print("\n" + "=" * 60)
    print("INVESTMENT COMMITTEE MEMO")
    print(result["ic_memo"])