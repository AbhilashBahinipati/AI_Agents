# Pattern 07: Plan and Execute
# Domain: Private Equity - Investment Due Diligence

from typing import TypedDict, List
import json
from langgraph.graph import StateGraph, END
from config import get_llm

llm = get_llm()


# -- State ------------------------------------------------------------------

class DDState(TypedDict):
    target_brief: str
    dd_plan: List[str]
    current_task_index: int
    completed_findings: List[str]
    final_report: str


# -- Planner Node -----------------------------------------------------------

def create_dd_plan(state: DDState) -> DDState:
    print("\n[Planner] Generating due diligence task plan...")
    prompt = f"""
    You are a due diligence planner at a private equity fund.
    Create a structured due diligence task list for this investment target.

    Generate 5 to 7 specific DD tasks tailored to this company's profile.
    Each task should be a concrete investigation with a clear deliverable.

    Return ONLY a JSON array of task strings. Example format:
    ["Analyze 3-year revenue trend and unit economics", "Review top 10 customer contracts for churn risk", ...]

    Target: {state['target_brief']}
    """
    # Planner must complete before execution begins — accumulate without streaming display
    response = ""
    for chunk in llm.stream(prompt):
        response += chunk.content

    try:
        raw = response.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        dd_plan = json.loads(raw.strip())
    except Exception:
        dd_plan = [
            line.strip('- "1234567890.[]')
            for line in response.split("\n")
            if line.strip() and len(line.strip()) > 15
        ][:7]

    print(f"Plan generated: {len(dd_plan)} tasks identified.")
    return {**state, "dd_plan": dd_plan, "current_task_index": 0, "completed_findings": []}


# -- Executor Node ----------------------------------------------------------

def execute_task(state: DDState) -> DDState:
    current_task = state["dd_plan"][state["current_task_index"]]
    task_num = state["current_task_index"] + 1
    total_tasks = len(state["dd_plan"])

    print(f"\n[Executor] Task {task_num} of {total_tasks}: {current_task[:70]}...")

    prompt = f"""
    You are a due diligence analyst at a private equity fund.
    Complete this due diligence task for the investment target.
    Use realistic financial figures and cite specific risks.

    Task: {current_task}
    Target: {state['target_brief']}

    Format your response as:
    TASK: {current_task}
    FINDING: [2 to 3 sentences of specific, data-driven findings]
    KEY RISK: [1 sentence describing the primary risk identified]
    RATING: Green / Amber / Red
    """
    response = ""
    for chunk in llm.stream(prompt):
        print(chunk.content, end="", flush=True)
        response += chunk.content
    print()

    updated_findings = state["completed_findings"] + [response]
    return {
        **state,
        "completed_findings": updated_findings,
        "current_task_index": state["current_task_index"] + 1
    }


def should_continue_execution(state: DDState) -> str:
    if state["current_task_index"] >= len(state["dd_plan"]):
        return "synthesize"
    return "execute"


# -- Synthesizer Node -------------------------------------------------------

def synthesize_dd_report(state: DDState) -> DDState:
    print("\n[Investment Director] Compiling final DD report...")
    findings_text = "\n\n---\n\n".join(state["completed_findings"])

    prompt = f"""
    You are an Investment Director at a private equity fund.
    Synthesize the due diligence findings into a final DD report
    for the Investment Committee.

    Structure:
    1. Investment Overview
    2. Overall DD Rating: Green (Proceed) / Amber (Proceed with Conditions) / Red (Do Not Invest)
    3. Findings Summary Table: Area | Rating | Key Finding
    4. Deal Breakers (if any)
    5. Conditions to Investment (if Amber)
    6. Recommended Next Steps

    Target: {state['target_brief']}

    DD Findings:
    {findings_text}
    """
    response = ""
    for chunk in llm.stream(prompt):
        print(chunk.content, end="", flush=True)
        response += chunk.content
    print()
    return {**state, "final_report": response}


# -- Graph ------------------------------------------------------------------

def build_graph():
    graph = StateGraph(DDState)

    graph.add_node("plan", create_dd_plan)
    graph.add_node("execute", execute_task)
    graph.add_node("synthesize", synthesize_dd_report)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "execute")
    graph.add_conditional_edges(
        "execute",
        should_continue_execution,
        {"execute": "execute", "synthesize": "synthesize"}
    )
    graph.add_edge("synthesize", END)

    return graph.compile()


# -- Run --------------------------------------------------------------------

if __name__ == "__main__":
    app = build_graph()

    target_brief = """
    Company: PayNxt Technologies Pvt. Ltd.
    Sector: B2B Fintech - Payments Infrastructure
    Stage: Series B, revenue-generating
    Business: Provides payment gateway and reconciliation software to mid-market
      retailers and logistics companies. 180 enterprise clients.
    Financials: FY24 Revenue Rs. 95 Cr (+68% YoY), EBITDA Rs. 8 Cr (8% margin, early stage).
      Monthly GMV processed: Rs. 4,200 Cr.
    Deal: PE fund evaluating 35% primary stake at Rs. 280 Cr valuation (3x Revenue).
    Key Concerns Flagged: Client concentration (top 5 clients = 48% revenue),
      RBI payment aggregator license pending renewal, founding team depth.
    """

    print("Plan and Execute - Private Equity Due Diligence")
    print("=" * 60)

    result = app.invoke({"target_brief": target_brief})

    print("\n" + "=" * 60)
    print(f"Due diligence complete. {len(result['dd_plan'])} tasks executed.")