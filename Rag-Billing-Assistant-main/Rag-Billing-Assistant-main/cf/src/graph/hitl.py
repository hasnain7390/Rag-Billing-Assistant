from __future__ import annotations

import re
import sys
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

# Ensure project root is importable when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.workflow import WorkflowState, build_workflow


def _extract_refund_details(query: str) -> tuple[str, float]:
    q = query.lower()
    if "enterprise" in q:
        plan = "Enterprise"
    elif "pro" in q:
        plan = "Pro"
    elif "basic" in q:
        plan = "Basic"
    else:
        plan = "Unknown"

    match = re.search(r"\$\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)", query)
    amount = 0.0
    if match:
        value = match.group(1) or match.group(2)
        try:
            amount = float(value)
        except ValueError:
            amount = 0.0

    return plan, amount


def build_hitl_graph():
    checkpointer = MemorySaver()
    # Pause execution right before refund logic so a manager can decide.
    return build_workflow(
        checkpointer=checkpointer,
        interrupt_before=["node_refund_logic"],
    )


def run_manager_cli() -> None:
    app = build_hitl_graph()

    thread_id = f"refund-thread-{uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    user_query = "I want a refund for my $200 Enterprise plan"
    initial_state: WorkflowState = {
        "messages": [HumanMessage(content=user_query)],
        "next_node": "node_router",
        "billing_context": [],
        "is_refund": False,
        "manager_decision": "",
        "refund_amount": 0.0,
    }

    print(f"Thread ID: {thread_id}")
    print(f"User request: {user_query}")

    # First run: should stop at interrupt_before node_refund_logic.
    app.invoke(initial_state, config=config)

    snapshot = app.get_state(config)
    next_nodes = list(snapshot.next or [])

    if "node_refund_logic" not in next_nodes:
        print("Did not pause at expected breakpoint. Current next nodes:", next_nodes)
        return

    print("\nGraph paused before node_refund_logic.")
    plan, amount = _extract_refund_details(user_query)
    print("Refund details:")
    print(f"- Plan: {plan}")
    print(f"- Refund Amount: ${amount:.2f}")

    while True:
        choice = input("Manager Approval Required. Type [A] to Approve, [R] to Reject: ").strip().upper()
        if choice in {"A", "R"}:
            break
        print("Invalid input. Please type A or R.")

    decision = "Approved" if choice == "A" else "Rejected"

    # Update checkpointed state for the same thread and resume execution.
    app.update_state(config, {"manager_decision": decision})
    final_state = app.invoke(None, config=config)

    print("\nFinal output:")
    print(final_state.get("answer", "No answer generated."))


if __name__ == "__main__":
    run_manager_cli()
