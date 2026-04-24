from __future__ import annotations

import json
import re
from typing import Any

from langchain_ollama import ChatOllama


ALLOWED_INTENTS = {"BILLING_QUERY", "REFUND_REQUEST", "OUT_OF_SCOPE"}


def _build_few_shot_prompt(query: str) -> str:
    return f"""
You are an intent classifier for a SaaS Billing and Refund Assistant.
Classify the user query into one of these categories:
- BILLING_QUERY: Questions about price, rules, policy, or general billing info.
- REFUND_REQUEST: Explicitly asking for money back or to cancel a paid subscription.
- OUT_OF_SCOPE: Anything not related to this SaaS billing/refund domain.

Few-shot examples:
User: "What is the price of the Enterprise plan?"
Output: {{"intent": "BILLING_QUERY", "confidence": 0.97}}

User: "Please refund my payment and cancel my Pro subscription."
Output: {{"intent": "REFUND_REQUEST", "confidence": 0.99}}

User: "Can you tell me today's cricket score?"
Output: {{"intent": "OUT_OF_SCOPE", "confidence": 0.98}}

Rules:
1) Return ONLY valid JSON with this exact schema:
   {{"intent": "CATEGORY", "confidence": 0.XX}}
2) intent must be one of: BILLING_QUERY, REFUND_REQUEST, OUT_OF_SCOPE
3) confidence must be a float between 0 and 1.
4) Do not include markdown, extra keys, or extra text.

User: "{query}"
""".strip()


def _coerce_result(data: dict[str, Any]) -> dict[str, Any]:
    intent = str(data.get("intent", "")).strip().upper()

    confidence = data.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    if intent not in ALLOWED_INTENTS:
        raise ValueError("Invalid intent category")

    return {"intent": intent, "confidence": confidence}


def _extract_json_object(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0)

    raise ValueError("No JSON object found in model output")


def _fallback_classify(query: str) -> dict[str, Any]:
    q = query.lower()

    # Required guardrail fallback from prompt: if "refund" appears, route to refund.
    if "refund" in q or "money back" in q or "cancel" in q:
        return {"intent": "REFUND_REQUEST", "confidence": 0.65}

    billing_keywords = [
        "billing",
        "price",
        "plan",
        "subscription",
        "payment",
        "failed payments",
        "enterprise",
        "pro",
        "basic",
        "policy",
        "rules",
    ]
    if any(keyword in q for keyword in billing_keywords):
        return {"intent": "BILLING_QUERY", "confidence": 0.6}

    return {"intent": "OUT_OF_SCOPE", "confidence": 0.55}


def classify_intent(query: str, model_name: str = "phi3:mini") -> dict[str, Any]:
    llm = ChatOllama(model=model_name, temperature=0)
    prompt = _build_few_shot_prompt(query)

    try:
        raw = llm.invoke(prompt).content
        if not isinstance(raw, str):
            raw = str(raw)
        json_text = _extract_json_object(raw)
        parsed = json.loads(json_text)
        return _coerce_result(parsed)
    except Exception:
        return _fallback_classify(query)


if __name__ == "__main__":
    test_queries = [
        "I want a refund for my Pro plan",
        "How many failed payments are allowed?",
        "What is the weather?",
    ]

    for q in test_queries:
        result = classify_intent(q)
        print(f"Query: {q}")
        print(f"Result: {json.dumps(result)}")
        print("-" * 60)
