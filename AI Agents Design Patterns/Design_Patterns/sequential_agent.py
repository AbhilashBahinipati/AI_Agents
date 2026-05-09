# Pattern 01: Sequential Pipeline
# Domain: Retail Banking - Loan Application Processing

from typing import TypedDict
from langgraph.graph import StateGraph, END
from config import get_llm

llm = get_llm()


# -- State ------------------------------------------------------------------

class LoanApplicationState(TypedDict):
    raw_application: str
    validated_application: str
    credit_report: str
    risk_score: str
    approval_decision: str
    offer_letter: str


# -- Nodes ------------------------------------------------------------------

def validate_application(state: LoanApplicationState) -> LoanApplicationState:
    print("\nValidating application...")
    prompt = f"""
    You are a loan application validator at a retail bank.
    Review the application and confirm it contains:
    - Applicant name and PAN number
    - Monthly income
    - Loan amount requested
    - Loan purpose
    - Employment type (salaried / self-employed)

    Application: {state['raw_application']}

    Output a clean validated application summary. Flag any missing fields.
    """
    response = ""
    for chunk in llm.stream(prompt):
        print(chunk.content, end="", flush=True)
        response += chunk.content
    print()
    return {**state, "validated_application": response}


def run_credit_check(state: LoanApplicationState) -> LoanApplicationState:
    print("\nRunning credit bureau check...")
    prompt = f"""
    You are a credit bureau analyst at a retail bank.
    Based on the validated application, generate a simulated credit report containing:
    - CIBIL score (between 300 and 900)
    - Number of active loans
    - Any defaults or late payments in the past 24 months
    - Total existing EMI obligations per month
    - Credit utilization ratio

    Validated Application: {state['validated_application']}
    """
    response = ""
    for chunk in llm.stream(prompt):
        print(chunk.content, end="", flush=True)
        response += chunk.content
    print()
    return {**state, "credit_report": response}


def calculate_risk_score(state: LoanApplicationState) -> LoanApplicationState:
    print("\nCalculating internal risk score...")
    prompt = f"""
    You are a credit risk analyst at a retail bank.
    Calculate an internal risk score (1 to 10, where 1 is lowest risk) based on:
    - CIBIL score from the credit report
    - Debt-to-income ratio (existing EMI + new EMI vs monthly income)
    - Employment stability
    - Loan-to-value considerations

    Provide a breakdown of the score with weightage for each factor.

    Credit Report: {state['credit_report']}
    Validated Application: {state['validated_application']}
    """
    response = ""
    for chunk in llm.stream(prompt):
        print(chunk.content, end="", flush=True)
        response += chunk.content
    print()
    return {**state, "risk_score": response}


def make_approval_decision(state: LoanApplicationState) -> LoanApplicationState:
    print("\nMaking approval decision...")
    prompt = f"""
    You are a loan approval officer at a retail bank.
    Based on the risk score and credit report, make a decision:
    - APPROVED: Risk score 1-4, CIBIL above 700
    - CONDITIONAL APPROVAL: Risk score 5-6, CIBIL 650-699 (specify conditions)
    - REJECTED: Risk score 7-10 or CIBIL below 650

    Provide a one-paragraph justification for the decision.

    Risk Score: {state['risk_score']}
    Credit Report: {state['credit_report']}
    """
    response = ""
    for chunk in llm.stream(prompt):
        print(chunk.content, end="", flush=True)
        response += chunk.content
    print()
    return {**state, "approval_decision": response}


def generate_offer_letter(state: LoanApplicationState) -> LoanApplicationState:
    print("\nGenerating offer letter...")
    prompt = f"""
    You are a loan operations officer at a retail bank.
    Based on the approval decision, generate a formal loan offer letter.
    If rejected, generate a formal decline letter with reason and next steps.

    Include:
    - Loan reference number (format: LN-2025-XXXXX)
    - Approved amount and interest rate (if approved)
    - Tenure options
    - EMI estimate
    - Next steps for the applicant

    Approval Decision: {state['approval_decision']}
    Validated Application: {state['validated_application']}
    """
    response = ""
    for chunk in llm.stream(prompt):
        print(chunk.content, end="", flush=True)
        response += chunk.content
    print()
    return {**state, "offer_letter": response}


# -- Graph ------------------------------------------------------------------

def build_graph():
    graph = StateGraph(LoanApplicationState)

    graph.add_node("validate", validate_application)
    graph.add_node("credit_check", run_credit_check)
    graph.add_node("risk_score", calculate_risk_score)
    graph.add_node("approval", make_approval_decision)
    graph.add_node("offer_letter", generate_offer_letter)

    graph.set_entry_point("validate")
    graph.add_edge("validate", "credit_check")
    graph.add_edge("credit_check", "risk_score")
    graph.add_edge("risk_score", "approval")
    graph.add_edge("approval", "offer_letter")
    graph.add_edge("offer_letter", END)

    return graph.compile()


# -- Run --------------------------------------------------------------------

if __name__ == "__main__":
    app = build_graph()

    sample_application = """
    Applicant: Rajesh Kumar Sharma
    PAN: ABCPK1234D
    Monthly Income: Rs. 1,20,000
    Loan Amount Requested: Rs. 25,00,000
    Loan Purpose: Home purchase
    Employment Type: Salaried
    Employer: Infosys Ltd
    Years of Employment: 6
    Existing EMI Obligations: Rs. 18,000 per month
    """

    print("Sequential Pipeline - Loan Application Processing")
    print("=" * 60)
    app.invoke({"raw_application": sample_application})
    print("\n" + "=" * 60)
    print("Pipeline complete.")