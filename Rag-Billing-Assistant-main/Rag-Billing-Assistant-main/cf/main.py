from __future__ import annotations

from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.graph.workflow import WorkflowState, build_workflow


def build_app():
    checkpointer = MemorySaver()
    return build_workflow(
        checkpointer=checkpointer,
        interrupt_before=["node_refund_logic"],
    )


def run_cli() -> None:
    app = build_app()
    thread_id = f"cli-thread-{uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    print("SaaS Billing & Refund Assistant (RAG + LangGraph + HITL)")
    print("Type your question. Type 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Assistant: Session ended.")
            break

        state: WorkflowState = {
            "messages": [HumanMessage(content=query)],
            "next_node": "node_router",
            "billing_context": [],
            "is_refund": False,
            "manager_decision": "",
            "refund_amount": 0.0,
        }

        # First execution pass. For refund requests, the graph pauses before refund logic.
        app.invoke(state, config=config)
        snapshot = app.get_state(config)
        next_nodes = list(snapshot.next or [])

        if "node_refund_logic" in next_nodes:
            print("\nManager Approval Required. Type [A] to Approve, [R] to Reject")
            while True:
                choice = input("Manager: ").strip().upper()
                if choice in {"A", "R"}:
                    break
                print("Please type A or R.")

            decision = "Approved" if choice == "A" else "Rejected"
            app.update_state(config, {"manager_decision": decision})
            final_state = app.invoke(None, config=config)
        else:
            final_state = snapshot.values

        intent = final_state.get("intent", "UNKNOWN")
        answer = final_state.get("answer", "I do not have a response.")

        print(f"Intent: {intent}")
        print(f"Assistant: {answer}\n")


if __name__ == "__main__":
    run_cli()
