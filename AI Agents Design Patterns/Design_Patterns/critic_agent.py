# Pattern 04: Reflection / Critic Loop
# Domain: Investment Banking - Equity Research Note

from typing import TypedDict
from langgraph.graph import StateGraph, END
from config import get_llm

llm = get_llm()
MAX_ITERATIONS = 3


# -- State ------------------------------------------------------------------

class ResearchNoteState(TypedDict):
    company_brief: str
    draft_note: str
    critique: str
    iteration: int
    approved: bool


# -- Nodes ------------------------------------------------------------------

def draft_research_note(state: ResearchNoteState) -> ResearchNoteState:
    iteration = state.get("iteration", 0) + 1
    print(f"\n[Analyst] Drafting research note - iteration {iteration}...")

    revision_context = ""
    if state.get("critique"):
        revision_context = f"\n\nPrevious critique to address in this revision:\n{state['critique']}"

    prompt = f"""
    You are a sell-side equity research analyst at an investment bank.
    Write a concise equity research note (maximum 350 words) with a
    BUY, HOLD, or SELL recommendation.

    Include:
    - Investment thesis (2 to 3 sentences, specific and data-driven)
    - Key financials: Revenue growth, EBITDA margin, P/E, ROE, Debt-to-Equity
    - Growth catalysts (at least 2 specific catalysts)
    - Key risks (at least 2 substantive risks)
    - 12-month target price with upside or downside percentage
    - Final recommendation with conviction level: High / Medium / Low

    Company Brief: {state['company_brief']}
    {revision_context}
    """
    response = ""
    for chunk in llm.stream(prompt):
        print(chunk.content, end="", flush=True)
        response += chunk.content
    print()
    return {**state, "draft_note": response, "iteration": iteration}


def critique_research_note(state: ResearchNoteState) -> ResearchNoteState:
    print(f"\n[Research Head] Reviewing draft...")

    prompt = f"""
    You are a research head at an investment bank reviewing an analyst's equity note.
    Apply the following quality checklist rigorously:

    1. Is the investment thesis specific and backed by data, not generic?
    2. Are actual financial metrics cited with numbers?
    3. Are growth catalysts company-specific, not sector platitudes?
    4. Are risk factors substantive and quantified where possible?
    5. Is the target price justified with a valuation methodology (P/E, DCF, EV/EBITDA)?
    6. Is the recommendation consistent with the analysis?

    If ALL six criteria are met, respond with exactly: APPROVED
    If not, respond with: REVISE - [specific numbered list of what needs fixing]

    Research Note:
    {state['draft_note']}
    """
    # Critic must complete fully before routing — accumulate without streaming display
    response = ""
    for chunk in llm.stream(prompt):
        response += chunk.content
    critique = response.strip()
    approved = critique.upper().startswith("APPROVED")
    print(f"Review outcome: {'APPROVED' if approved else 'REVISE'}")
    if not approved:
        print(f"Critique: {critique}")
    return {**state, "critique": critique, "approved": approved}


def should_continue(state: ResearchNoteState) -> str:
    if state["approved"]:
        return "done"
    if state["iteration"] >= MAX_ITERATIONS:
        return "done"
    return "revise"


# -- Graph ------------------------------------------------------------------

def build_graph():
    graph = StateGraph(ResearchNoteState)

    graph.add_node("draft", draft_research_note)
    graph.add_node("critique", critique_research_note)

    graph.set_entry_point("draft")
    graph.add_edge("draft", "critique")
    graph.add_conditional_edges(
        "critique",
        should_continue,
        {"revise": "draft", "done": END}
    )

    return graph.compile()


# -- Run --------------------------------------------------------------------

if __name__ == "__main__":
    app = build_graph()

    company_brief = """
    Company: Bajaj Finance Ltd (NSE: BAJFINANCE)
    Sector: Non-Banking Financial Company (NBFC), Consumer and SME Lending
    Context: Largest consumer NBFC in India. AUM of Rs. 3.5 lakh crore.
    Recent Results: Q3FY25 Net Interest Income Rs. 9,800 Cr (+28% YoY),
    PAT Rs. 4,100 Cr (+22% YoY). Gross NPA at 0.85%.
    Concern: Rising cost of funds as RBI holds rates. Unsecured loan mix at 32%.
    CMP: Rs. 7,200 | 52W High: Rs. 8,050 | 52W Low: Rs. 6,200
    """

    print("Reflection Loop - Equity Research Note")
    print("=" * 60)

    result = app.invoke({
        "company_brief": company_brief,
        "draft_note": "",
        "critique": "",
        "iteration": 0,
        "approved": False
    })

    print("\n" + "=" * 60)
    status = "APPROVED" if result["approved"] else "MAX ITERATIONS REACHED - NOT APPROVED"
    print(f"Final status: {status} after {result['iteration']} iteration(s).")