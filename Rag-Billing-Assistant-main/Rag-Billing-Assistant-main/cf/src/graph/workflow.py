from __future__ import annotations

import re
from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.graph.router import classify_intent
from src.rag.chain import build_rag_chain


class WorkflowState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    next_node: str
    billing_context: List[str]
    is_refund: bool
    requires_approval: bool
    refund_amount: float
    manager_decision: str
    intent: str
    confidence: float
    answer: str


_RAG_CHAIN = None


def _get_rag_chain():
    global _RAG_CHAIN
    if _RAG_CHAIN is None:
        _RAG_CHAIN = build_rag_chain(model_name="phi3:mini", k=3)
    return _RAG_CHAIN


def _latest_user_query(state: WorkflowState) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    if state.get("messages"):
        return str(state["messages"][-1].content)
    return ""


def _extract_refund_amount(query: str) -> float:
    match = re.search(r"\$\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)", query)
    if not match:
        return 0.0
    val = match.group(1) or match.group(2)
    try:
        return float(val)
    except ValueError:
        return 0.0


def node_router(state: WorkflowState) -> WorkflowState:
    query = _latest_user_query(state)
    result = classify_intent(query)
    intent = result["intent"]

    if intent == "BILLING_QUERY":
        next_node = "node_billing_rag"
    elif intent == "REFUND_REQUEST":
        next_node = "node_refund_logic"
    else:
        next_node = "node_out_of_scope"

    return {
        "intent": intent,
        "confidence": float(result.get("confidence", 0.0)),
        "is_refund": intent == "REFUND_REQUEST",
        "next_node": next_node,
        "messages": [AIMessage(content=f"Router intent={intent} confidence={result.get('confidence', 0.0)}")],
    }


def node_billing_rag(state: WorkflowState) -> WorkflowState:
    query = _latest_user_query(state)
    rag_chain = _get_rag_chain()
    result = rag_chain.invoke({"input": query})

    answer = str(result.get("answer", "I do not have that information. Please contact manager@saas.com."))
    context_docs = result.get("context", [])
    snippets = [doc.page_content.strip().replace("\n", " ")[:220] for doc in context_docs]

    return {
        "answer": answer,
        "billing_context": snippets,
        "next_node": "done",
        "messages": [AIMessage(content=answer)],
    }


def node_refund_logic(state: WorkflowState) -> WorkflowState:
    query = _latest_user_query(state).lower()
    amount = _extract_refund_amount(query)

    is_pro = "pro" in query
    is_enterprise = "enterprise" in query
    requires_approval = is_pro or is_enterprise
    manager_decision = str(state.get("manager_decision", "")).strip().capitalize()

    if requires_approval:
        if manager_decision not in {"Approved", "Rejected"}:
            manager_decision = "Rejected"
        answer = f"A manager has {manager_decision} your refund based on the policy."
    else:
        manager_decision = "Approved"
        answer = "A manager has Approved your refund based on the policy."

    return {
        "is_refund": True,
        "refund_amount": amount,
        "requires_approval": requires_approval,
        "manager_decision": manager_decision,
        "answer": answer,
        "next_node": "done",
        "messages": [AIMessage(content=answer)],
    }


def node_out_of_scope(state: WorkflowState) -> WorkflowState:
    answer = "I only handle SaaS billing and refund related questions."
    return {
        "answer": answer,
        "next_node": "done",
        "messages": [AIMessage(content=answer)],
    }


def _route_from_intent(state: WorkflowState) -> str:
    intent = state.get("intent", "OUT_OF_SCOPE")
    if intent == "BILLING_QUERY":
        return "node_billing_rag"
    if intent == "REFUND_REQUEST":
        return "node_refund_logic"
    return "node_out_of_scope"


def build_workflow(checkpointer=None, interrupt_before: list[str] | None = None):
    graph = StateGraph(WorkflowState)

    graph.add_node("node_router", node_router)
    graph.add_node("node_billing_rag", node_billing_rag)
    graph.add_node("node_refund_logic", node_refund_logic)
    graph.add_node("node_out_of_scope", node_out_of_scope)

    graph.add_edge(START, "node_router")
    graph.add_conditional_edges(
        "node_router",
        _route_from_intent,
        {
            "node_billing_rag": "node_billing_rag",
            "node_refund_logic": "node_refund_logic",
            "node_out_of_scope": "node_out_of_scope",
        },
    )
    graph.add_edge("node_billing_rag", END)
    graph.add_edge("node_refund_logic", END)
    graph.add_edge("node_out_of_scope", END)

    return graph.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)


if __name__ == "__main__":
    app = build_workflow()

    # Billing Query path smoke test: Router -> billing_rag
    initial_state: WorkflowState = {
        "messages": [HumanMessage(content="How many failed payments are allowed?")],
        "next_node": "node_router",
        "billing_context": [],
        "is_refund": False,
    }

    final_state = app.invoke(initial_state)

    print("\n=== Billing Path Test ===")
    print("Intent:", final_state.get("intent"))
    print("Next node:", final_state.get("next_node"))
    print("Is refund:", final_state.get("is_refund"))
    print("Answer:", final_state.get("answer"))

    print("\nRetrieved billing_context snippets:")
    for i, snippet in enumerate(final_state.get("billing_context", []), start=1):
        print(f"{i}. {snippet}")
