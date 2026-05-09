# Pattern 03: Router / Conditional Branching
# Domain: Corporate Banking - Client Query Triage

from typing import TypedDict
from langgraph.graph import StateGraph, END
from config import get_llm

llm = get_llm()


# -- State ------------------------------------------------------------------

class QueryState(TypedDict):
    client_query: str
    query_type: str
    response: str
    escalation_needed: bool


# -- Router Node ------------------------------------------------------------

def classify_query(state: QueryState) -> QueryState:
    print("\nClassifying client query...")
    prompt = f"""
    You are a corporate banking query classifier.
    Classify this client query into exactly one of these categories:
    TRANSACTION | CREDIT | COMPLIANCE | GENERAL

    Rules:
    - TRANSACTION: Payment failures, fund transfers, reconciliation issues, SWIFT queries
    - CREDIT: Working capital limits, loan drawdowns, interest rate queries, LC/BG requests
    - COMPLIANCE: KYC renewal, AML flagging, regulatory reporting, FEMA queries
    - GENERAL: Account servicing, RM contact, product information, anything else

    Client Query: {state['client_query']}

    Respond with ONLY the category word. Nothing else.
    """
    # Classifier must complete fully before routing — accumulate without streaming display
    response = ""
    for chunk in llm.stream(prompt):
        response += chunk.content
    query_type = response.strip().upper()
    if query_type not in ["TRANSACTION", "CREDIT", "COMPLIANCE", "GENERAL"]:
        query_type = "GENERAL"
    print(f"Query classified as: {query_type}")
    return {**state, "query_type": query_type}


# -- Routing Logic ----------------------------------------------------------

def route_query(state: QueryState) -> str:
    return state["query_type"].lower()


# -- Specialist Nodes -------------------------------------------------------

def handle_transaction(state: QueryState) -> QueryState:
    print("\n[Transaction Specialist] Preparing response...")
    prompt = f"""
    You are a transaction banking specialist at a corporate bank.
    Resolve this transaction-related query:
    1. Likely root cause
    2. Immediate resolution steps
    3. Expected timeline for resolution
    4. Preventive recommendation

    Query: {state['client_query']}
    """
    response = ""
    for chunk in llm.stream(prompt):
        print(chunk.content, end="", flush=True)
        response += chunk.content
    print()
    return {**state, "response": response, "escalation_needed": False}


def handle_credit(state: QueryState) -> QueryState:
    print("\n[Credit Specialist] Preparing response...")
    prompt = f"""
    You are a credit specialist at a corporate bank.
    Address this credit or lending query:
    1. Assessment of the request
    2. Required documentation
    3. Applicable process and timeline
    4. Any regulatory or policy considerations

    Query: {state['client_query']}
    """
    response = ""
    for chunk in llm.stream(prompt):
        print(chunk.content, end="", flush=True)
        response += chunk.content
    print()
    return {**state, "response": response, "escalation_needed": False}


def handle_compliance(state: QueryState) -> QueryState:
    print("\n[Compliance Officer] Preparing response...")
    prompt = f"""
    You are a compliance officer at a corporate bank.
    Address this compliance or regulatory query:
    1. Regulatory framework applicable
    2. Required actions from the client
    3. Bank's obligations and timeline
    4. Risk of non-compliance

    Query: {state['client_query']}
    """
    response = ""
    for chunk in llm.stream(prompt):
        print(chunk.content, end="", flush=True)
        response += chunk.content
    print()
    return {**state, "response": response, "escalation_needed": True}


def handle_general(state: QueryState) -> QueryState:
    print("\n[Relationship Manager] Preparing response...")
    prompt = f"""
    You are a relationship manager at a corporate bank.
    Address this general client query professionally.
    If it requires specialist intervention, say so clearly.

    Query: {state['client_query']}
    """
    response = ""
    for chunk in llm.stream(prompt):
        print(chunk.content, end="", flush=True)
        response += chunk.content
    print()
    return {**state, "response": response, "escalation_needed": False}


# -- Graph ------------------------------------------------------------------

def build_graph():
    graph = StateGraph(QueryState)

    graph.add_node("classify", classify_query)
    graph.add_node("transaction", handle_transaction)
    graph.add_node("credit", handle_credit)
    graph.add_node("compliance", handle_compliance)
    graph.add_node("general", handle_general)

    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify",
        route_query,
        {
            "transaction": "transaction",
            "credit": "credit",
            "compliance": "compliance",
            "general": "general",
        }
    )
    graph.add_edge("transaction", END)
    graph.add_edge("credit", END)
    graph.add_edge("compliance", END)
    graph.add_edge("general", END)

    return graph.compile()


# -- Run --------------------------------------------------------------------

if __name__ == "__main__":
    app = build_graph()

    queries = [
        "Our SWIFT payment of USD 5,00,000 to our Singapore subsidiary sent yesterday has not been credited. UETR: 12345-ABCDE. Please investigate urgently.",
        "We need to increase our working capital limit from Rs. 50 Cr to Rs. 75 Cr ahead of the festive season. What documents are required?",
        "We received an AML flag on our account last week. Our legal team needs to understand the specific transactions flagged and the bank's next steps.",
    ]

    for query in queries:
        print("\n" + "=" * 60)
        print(f"CLIENT QUERY: {query[:80]}...")
        result = app.invoke({"client_query": query, "escalation_needed": False})
        print(f"\nESCALATION REQUIRED: {result['escalation_needed']}")