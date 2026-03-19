# Agentic AI Design Patterns — Comprehensive Guide

This guide covers the 7 core agentic AI design patterns implemented in this repository. For each pattern you will find: what it is, how it works, the real-world use case, when to use it, when to avoid it, and the key tradeoffs. The final section is a decision framework to help you pick the right pattern for your problem.

---

## Table of Contents

1. [Pattern 01 — Sequential Pipeline](#pattern-01--sequential-pipeline)
2. [Pattern 02 — Parallel Fan-out / Fan-in](#pattern-02--parallel-fan-out--fan-in)
3. [Pattern 03 — Router / Conditional Branching](#pattern-03--router--conditional-branching)
4. [Pattern 04 — Reflection / Critic Loop](#pattern-04--reflection--critic-loop)
5. [Pattern 05 — Hierarchical Supervisor](#pattern-05--hierarchical-supervisor)
6. [Pattern 06 — Human-in-the-Loop](#pattern-06--human-in-the-loop)
7. [Pattern 07 — Plan and Execute](#pattern-07--plan-and-execute)
8. [Pattern Selection Framework](#pattern-selection-framework)
9. [Combining Patterns](#combining-patterns)
10. [Common Mistakes](#common-mistakes)

---

## Pattern 01 — Sequential Pipeline

### What it is

A fixed, ordered chain of agent nodes where the output of each step becomes the input to the next. No branching, no parallelism, no loops. The simplest agentic pattern and the right starting point for most deterministic workflows.

### How it works

```
Input → Node A → Node B → Node C → Node D → Output
```

Each node receives the full accumulated state, adds its output to that state, and passes it forward. The graph is a straight line.

### Use case: Retail Banking — Loan Application Processing

A home loan application passes through five sequential steps:

1. **Validation** — Confirm all required fields are present (PAN, income, loan amount, employment type)
2. **Credit Check** — Pull CIBIL score, active loans, defaults, EMI obligations
3. **Risk Scoring** — Calculate internal risk score (1–10) based on DTI ratio, CIBIL, and employment stability
4. **Approval Decision** — Apply policy rules: Approved / Conditional / Rejected
5. **Offer Letter** — Generate formal communication with loan reference, EMI, and tenure options

Each step is strictly dependent on the previous. The credit check cannot run before validation confirms the applicant exists. The approval decision cannot be made without the risk score. This dependency chain is the defining characteristic of the sequential pattern.

### When to use

- Steps have strict output-to-input dependencies
- The process is deterministic and repeatable
- Auditability is important — you need a clear trace of every intermediate step
- The task maps naturally to a business process or SOP

### When to avoid

- Steps are independent of each other — use Parallel Fan-out instead
- The input type determines which steps to run — use Router or Plan and Execute instead
- The output needs iterative improvement — use Reflection instead

### Tradeoffs

| Dimension | Assessment |
|-----------|-----------|
| Complexity | Low — easiest pattern to implement and debug |
| Latency | Proportional to number of steps, no parallelism |
| Cost | Low — one LLM call per step, no loops |
| Flexibility | None — graph is fixed at design time |
| Auditability | High — every intermediate state is captured |

---

## Pattern 02 — Parallel Fan-out / Fan-in

### What it is

A single input is broadcast to multiple independent agent nodes that run simultaneously (fan-out), and their outputs are merged before being passed to a synthesis node (fan-in). Reduces wall-clock time for workloads with independent subtasks.

### How it works

```
                 ┌→ Specialist A ─┐
Input → Fan-out ─├→ Specialist B ─├→ Fan-in → Synthesis → Output
                 └→ Specialist C ─┘
```

In LangGraph, the `Send` API dispatches to multiple nodes in a single step. Each parallel node must return **only the key it owns** — returning `{**state, ...}` causes `InvalidUpdateError` when two nodes attempt to write to the same shared key concurrently.

### Use case: Wealth Management — Portfolio Risk Analysis

A client's Rs. 2 crore portfolio requires three independent risk analyses before the Relationship Manager can act. These analyses have no dependency on each other — running them sequentially would triple the latency for no reason.

Three specialist agents run in parallel:

- **Market Risk Agent** — Beta, VaR at 95%, drawdown exposure by sector
- **Liquidity Risk Agent** — Illiquid holdings, percentage liquidatable within 3 days
- **Concentration Risk Agent** — Top holding weights, sector concentration above 30%, diversification gaps

A synthesis agent combines all three into a single Risk Summary Report with an overall portfolio risk score.

### When to use

- Multiple independent analyses are required on the same input
- Latency is a concern — parallel execution reduces total time proportionally
- The final output requires synthesis of multiple expert perspectives
- Subtasks do not depend on each other's outputs

### When to avoid

- Tasks have sequential dependencies (use Sequential)
- Token budget is constrained — parallel means multiple simultaneous LLM calls
- The number of parallel branches is dynamic (use Plan and Execute instead)

### Tradeoffs

| Dimension | Assessment |
|-----------|-----------|
| Complexity | Low-Medium — fan-out logic is straightforward, but return key discipline is required |
| Latency | Low — wall-clock time equals the slowest branch, not the sum |
| Cost | Medium — all branches call the LLM simultaneously |
| Flexibility | Low — branches are fixed at design time |
| Auditability | High — each branch result is captured separately in state |

---

## Pattern 03 — Router / Conditional Branching

### What it is

An input is first classified by a router node, which then conditionally routes execution to one of several specialist handlers. Only one path executes per invocation. The router is the traffic controller — it reads the input, decides the destination, and dispatches accordingly.

### How it works

```
Input → Classifier → [condition] → Specialist A
                                 → Specialist B
                                 → Specialist C
                                 → Fallback Handler
```

### Use case: Corporate Banking — Client Query Triage

A corporate bank receives hundreds of inbound queries daily across very different domains. A single monolithic agent handling all query types performs poorly — a SWIFT payment specialist is not the right agent for a KYC compliance query.

The router classifies each query into one of four categories and dispatches to the appropriate specialist:

- **TRANSACTION** — Payment failures, SWIFT issues, reconciliation, fund transfer disputes
- **CREDIT** — Working capital limit increases, loan drawdowns, LC/BG requests, interest rate queries
- **COMPLIANCE** — KYC renewal, AML flags, FEMA queries, regulatory reporting obligations
- **GENERAL** — Account servicing, RM contact, product information

Each specialist agent is prompt-engineered for its domain, producing better responses than a general-purpose agent attempting all four.

### When to use

- Input type or content determines which handling logic applies
- Specialist agents outperform generalist agents in each domain
- You want to avoid a single monolithic prompt doing everything
- Categories are well-defined with minimal overlap

### When to avoid

- Categories heavily overlap — classification accuracy degrades and routing errors compound
- All inputs require the same treatment — unnecessary overhead, use Sequential
- The number of categories grows beyond 6–8 — maintainability suffers

### Tradeoffs

| Dimension | Assessment |
|-----------|-----------|
| Complexity | Medium — requires reliable classifier prompt engineering |
| Latency | Low — only one branch executes |
| Cost | Low — classifier call plus one specialist call |
| Flexibility | Medium — adding new categories requires adding new nodes |
| Auditability | Medium — classification decision is visible in state |

---

## Pattern 04 — Reflection / Critic Loop

### What it is

A writer agent produces an initial output. A separate critic agent evaluates it against a defined quality rubric and either approves it or returns it to the writer with specific feedback. This loop continues until the output meets the standard or a maximum iteration count is reached.

### How it works

```
Input → Writer → Critic → [approved?] → YES → Output
                    ↑                 → NO  → Writer (with critique)
                    └─────────────────────────┘
```

### Use case: Investment Banking — Equity Research Note

Equity research notes go to institutional investors and carry reputational and regulatory weight. A single-pass AI generation is often too generic — vague investment theses, missing actual numbers, unsubstantiated price targets.

The reflection loop enforces a quality standard:

- **Writer Agent** — Drafts a BUY/HOLD/SELL note with thesis, financials, catalysts, risks, and target price
- **Critic Agent** — Checks against a six-point rubric: Is the thesis data-driven? Are actual metrics cited? Are catalysts company-specific? Are risks quantified? Is the target price methodology stated? Is the recommendation consistent?

The critic returns either `APPROVED` or `REVISE - [numbered list of issues]`. The writer addresses each issue in the next iteration. Maximum 3 iterations before the loop exits regardless.

### When to use

- Output quality is critical and a single pass is not reliable enough
- Quality criteria can be defined explicitly and checked programmatically
- The domain has a reviewable standard (financial analysis, legal drafting, medical documentation, code review)
- Iteration count can be bounded to control cost

### When to avoid

- Real-time or low-latency response is required — each loop adds a full round-trip
- Quality criteria are subjective or not definable upfront
- The cost of multiple LLM calls per output is prohibitive at scale

### Tradeoffs

| Dimension | Assessment |
|-----------|-----------|
| Complexity | Medium — requires a well-designed critic prompt |
| Latency | High — multiplies with each iteration |
| Cost | High — each iteration is 2 LLM calls (writer + critic) |
| Output Quality | High — iterative refinement reliably improves quality |
| Flexibility | Low — quality criteria are hardcoded in the critic |

---

## Pattern 05 — Hierarchical Supervisor

### What it is

A supervisor agent decomposes a complex task and delegates to multiple specialist sub-agents, each responsible for a distinct domain. The supervisor coordinates the workflow — planning at the start, synthesizing at the end. Sub-agents operate independently and report back their findings.

### How it works

```
                      ┌→ Specialist A ─┐
Input → Supervisor ───├→ Specialist B ─├→ Supervisor Synthesize → Output
        (planning)    ├→ Specialist C ─┘      (IC memo)
                      └→ Specialist D ─┘
```

The supervisor adds a coordination layer that the pure parallel pattern lacks — it creates a plan before dispatch and synthesizes intelligently after, rather than just merging raw outputs.

### Use case: Investment Banking — M&A Deal Coordination

An M&A transaction involves fundamentally different domains of expertise that must work together toward a single recommendation. No single agent can credibly handle all of them.

- **Supervisor (Planning)** — Reviews the deal brief and defines priorities for each workstream
- **Valuation Agent** — DCF, comparable companies, precedent transactions, offer price range
- **Legal Agent** — CCI filing requirements, FEMA/RBI approvals, key representations, DD checklist
- **Synergy Agent** — Revenue and cost synergies with Year 1/2/3 timeline and NPV
- **Financing Agent** — Equity/debt split, instruments, pro forma leverage, debt serviceability
- **Supervisor (Synthesis)** — Produces the Investment Committee memo with a PROCEED / PROCEED WITH CONDITIONS / DO NOT PROCEED recommendation

### When to use

- The task is genuinely multi-domain and too complex for a single agent
- The right decomposition is dynamic — it depends on the specific input
- Accountability and coordination between specialists matters
- The final output requires intelligent synthesis, not just concatenation

### When to avoid

- The task is straightforward — a supervisor layer adds overhead with no benefit
- Subtasks are always identical — use Parallel Fan-out instead (simpler, same result)
- Latency constraints are very tight — the supervisor adds two sequential LLM calls (plan and synthesize) around the parallel execution

### Tradeoffs

| Dimension | Assessment |
|-----------|-----------|
| Complexity | High — requires careful prompt engineering for both supervisor and sub-agents |
| Latency | Medium — parallel sub-agents, but supervisor adds sequential bookend calls |
| Cost | High — supervisor (x2) plus all sub-agents |
| Flexibility | Medium — sub-agent roster is fixed, supervisor prompt can adapt |
| Output Quality | High — coordination and synthesis produce coherent, structured outputs |

---

## Pattern 06 — Human-in-the-Loop

### What it is

AI agents handle all the analytical groundwork — research, recommendation, risk checks — and then pause execution at a defined checkpoint. A human reviews the output and provides a decision: approve, reject, or modify. Execution continues only after the human responds. The AI cannot proceed unilaterally.

### How it works

```
Input → AI Recommendation → Risk Check → [PAUSE: Human Decision]
                                                    ↓
                                         APPROVE → Execute
                                         REJECT  → End
                                         MODIFY  → Revise → Execute
```

LangGraph's `MemorySaver` (or a persistent checkpointer in production) saves state at the pause point so the graph can resume after an arbitrarily long human delay without losing context.

### Use case: Wealth Management — High-Value Trade Approval

A trade recommendation for an HNI client involves substantial capital. Regulatory expectations (SEBI suitability norms), fiduciary responsibility, and the irreversibility of trade execution all require a human in the decision chain.

- **AI Recommendation Agent** — Analyzes market signal against client profile, recommends specific trade (asset, quantity, price)
- **Risk Check Agent** — Validates trade against guardrails: single trade below 10% of AUM, sector concentration below 35%, no derivatives for this client category
- **Human Checkpoint** — Relationship Manager reviews the recommendation and risk check, enters APPROVE / REJECT / MODIFY
- **Execution Agent** — Generates trade confirmation with reference number, filled price, charges, and settlement date

### When to use

- The decision is irreversible once executed (trade, disbursement, deletion, legal filing)
- Regulatory or compliance frameworks require documented human authorization
- Errors carry high financial, legal, or reputational consequences
- The organization is in an early phase of AI adoption and trust must be built incrementally

### When to avoid

- High-volume, low-stakes decisions where human review creates an unacceptable bottleneck
- The AI's reliability on this specific decision type is already validated to a high standard
- Real-time execution is required with no tolerance for human latency

### Tradeoffs

| Dimension | Assessment |
|-----------|-----------|
| Complexity | High — requires state persistence and interrupt/resume infrastructure |
| Latency | Very High — bounded by human response time |
| Cost | Medium — AI calls are few; human time is the real cost |
| Risk | Low — human accountability is embedded in the workflow |
| Scalability | Low — throughput is limited by human availability |

---

## Pattern 07 — Plan and Execute

### What it is

A planner agent first analyses the problem and dynamically generates a task list specific to that input. An executor agent then works through each task sequentially, accumulating findings. A synthesizer compiles everything into a final output. The key distinction from Sequential is that the steps are not known at design time — they are generated at runtime by the planner.

### How it works

```
Input → Planner (generates N tasks) → Executor (task 1) → Executor (task 2) → ... → Executor (task N) → Synthesize → Output
```

The executor loop runs until all tasks are complete, using a conditional edge to check whether the current task index has reached the end of the plan.

### Use case: Private Equity — Investment Due Diligence

A fintech investment target (PayNxt Technologies, a B2B payments company at Series B) requires due diligence. But the right DD tasks for a payments infrastructure company are different from those for a lending company or an insurance distributor. A fixed sequential pipeline would ask the same questions regardless of the target — missing the nuance.

The planner generates a custom task list for PayNxt based on its specific profile: RBI payment aggregator license risk, client concentration in top 5 accounts, unit economics on GMV, founding team depth. The executor investigates each task and produces structured findings with a Red/Amber/Green rating. The synthesizer produces an IC-ready DD report with a PROCEED / PROCEED WITH CONDITIONS / DO NOT INVEST recommendation.

### When to use

- The right subtasks cannot be determined without first understanding the specific input
- Different inputs require fundamentally different investigation approaches
- The problem is open-ended — research, discovery, or exploratory analysis
- You need the agent to demonstrate reasoning about what to investigate, not just execute a fixed checklist

### When to avoid

- The task is well-defined and the steps are always the same — use Sequential (simpler, cheaper, faster)
- Budget is very tight — this is the most expensive pattern (planner call + N executor calls + synthesizer call)
- Real-time response is required — planning adds latency before any execution begins

### Tradeoffs

| Dimension | Assessment |
|-----------|-----------|
| Complexity | Very High — planner, executor loop, and synthesizer all require careful design |
| Latency | Very High — proportional to number of planned tasks |
| Cost | Very High — most LLM calls of any pattern |
| Flexibility | Very High — plan adapts completely to each input |
| Output Quality | High — depth of investigation is proportional to task count |

---

## Pattern Selection Framework

### Decision Tree

Work through these questions in order. Stop at the first match.

```
1. Are the steps fixed and known upfront, with strict sequential dependencies?
   YES → Sequential Pipeline (Pattern 01)

2. Are the subtasks independent and can run simultaneously?
   YES → Parallel Fan-out (Pattern 02)

3. Does the input type or content determine which handling path to take?
   YES → Router (Pattern 03)

4. Does the output need iterative self-improvement through a critic?
   YES → Reflection Loop (Pattern 04)

5. Is the task multi-domain, requiring coordination across specialist agents?
   YES → Hierarchical Supervisor (Pattern 05)

6. Is the decision irreversible, regulated, or does it require human accountability?
   YES → Human-in-the-Loop (Pattern 06)

7. Are the right subtasks unknown upfront and must be determined dynamically?
   YES → Plan and Execute (Pattern 07)
```

### Comparison Matrix

| Dimension | Sequential | Parallel | Router | Reflection | Hierarchical | HITL | Plan & Execute |
|-----------|-----------|----------|--------|-----------|--------------|------|----------------|
| Steps known upfront | Yes | Yes | Partial | Yes | Partial | Yes | No |
| Human involvement | None | None | None | None | None | Required | Optional |
| Latency | Medium | Low | Low | High | Medium | Very High | Very High |
| LLM call cost | Low | Medium | Low | High | High | Medium | Very High |
| Output adaptability | None | None | Medium | Low | Medium | Low | Very High |
| Implementation complexity | Low | Low-Med | Medium | Medium | High | High | Very High |
| Best starting point | Yes | Yes | Yes | No | No | No | No |

### Quick Reference

```
Fixed steps, one after another              →  Sequential
Same input, multiple analyses at once       →  Parallel
Input type determines the path              →  Router
Output needs iterative improvement          →  Reflection
Multi-domain task needs coordination        →  Hierarchical
Human must authorize the decision           →  Human-in-the-Loop
Right steps unknown until you see the input →  Plan and Execute
```

### By Financial Services Domain

| Domain | Recommended Patterns | Rationale |
|--------|---------------------|-----------|
| Retail Banking | Sequential, Router | Standardized processes, high volume, rule-based routing |
| Corporate Banking | Router, HITL | Complex query types, large transaction values need approval |
| Wealth Management | Parallel, HITL | Multi-dimensional analysis, high-stakes client decisions |
| Investment Banking | Hierarchical, Reflection | Multi-workstream deals, quality-critical deliverables |
| Private Equity | Plan and Execute, Reflection | Open-ended research, high-quality output requirements |
| Risk and Compliance | Sequential, HITL | Regulated processes, mandatory human oversight |
| Treasury | Sequential, Parallel | Deterministic workflows, simultaneous market data analysis |

---

## Combining Patterns

Production systems rarely use a single pattern in isolation. The patterns are composable — you can nest them or chain them to handle more complex workflows.

**Router → Sequential**
Route the input to the right pipeline. A customer query is first classified (Router), then the appropriate processing pipeline runs (Sequential). Used in retail banking servicing systems.

**Hierarchical + Parallel**
The supervisor dispatches to sub-agents that run in parallel, then synthesizes. Reduces total latency of the hierarchical pattern significantly. Applicable in M&A situations where all workstreams can start simultaneously.

**Reflection + Hierarchical**
Each sub-agent in a hierarchical system runs its own internal reflection loop before reporting to the supervisor. Higher quality outputs at higher cost. Appropriate for high-stakes deliverables like credit committee reports.

**Plan and Execute + Human-in-the-Loop**
The planner generates a task list, a human reviews and approves the plan before execution begins, then the executor proceeds. Useful for sensitive or regulated investigations (forensic audit, litigation support).

**Sequential → Human-in-the-Loop**
A pipeline runs all analytical steps, then pauses for human authorization before the final irreversible action. Standard pattern for loan disbursement: underwriting pipeline runs fully, credit officer approves, then disbursement executes.

---

## Common Mistakes

**Choosing Plan and Execute when Sequential would do**
If the steps are always the same, you do not need a planner. Plan and Execute is significantly more expensive and slower. Start with Sequential and upgrade only when the fixed steps demonstrably fail for certain inputs.

**Using {**state} returns in parallel nodes**
Any node running concurrently with another must return only the keys it writes. Returning the full state causes `InvalidUpdateError` in LangGraph. This applies to both Pattern 02 (Parallel Fan-out) and Pattern 05 (Hierarchical sub-agents).

**Building a Hierarchical system when Parallel is sufficient**
If the sub-tasks are always the same and do not need a supervisor to coordinate or plan, Parallel Fan-out is simpler and achieves the same result with fewer LLM calls.

**Unbounded Reflection loops**
Always set a `MAX_ITERATIONS` cap on reflection loops. Without it, a pathological critic prompt can loop indefinitely. Three iterations is sufficient for most use cases — quality gains beyond that are marginal.

**Skipping Human-in-the-Loop for regulated decisions**
In financial services, SEBI, RBI, and internal compliance frameworks often mandate human authorization for specific transaction types. Building a fully automated pipeline for these decisions creates regulatory and operational risk regardless of AI accuracy.

**Using MemorySaver in production HITL systems**
`MemorySaver` stores state in RAM. A process restart loses all checkpoints. Use `SqliteSaver` for single-instance deployments or `PostgresSaver` for distributed systems when building production HITL workflows.