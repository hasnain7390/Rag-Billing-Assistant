# 🧠 RAG-Based SaaS Billing & Refund Assistant (LangGraph + HITL)

## 🚀 Overview

This project is an **AI-powered assistant** designed to handle SaaS billing queries using:

- 🔍 Retrieval-Augmented Generation (RAG)  
- 🔄 LangGraph workflow orchestration  
- 🛑 Human-in-the-Loop (HITL) decision system  

The system automates **~80% of customer queries** while ensuring **100% safety for financial decisions** by requiring human approval for sensitive actions like refunds.

---

## 🎯 Problem Statement

Traditional customer support systems:

- ❌ Repetitive and time-consuming  
- ❌ Prone to LLM hallucinations  
- ❌ Unsafe for financial decisions  

This system solves these issues by:

- ✅ Grounding responses using policy documents  
- ✅ Separating retrieval from decision-making  
- ✅ Adding human approval for critical actions  

---

## 🧠 Key Features

- 📄 PDF-based knowledge retrieval (Billing Policy)  
- 🔍 Semantic search using embeddings (ChromaDB)  
- 🧠 Intent classification (Billing / Refund / Out-of-Scope)  
- 🔁 Graph-based workflow using LangGraph  
- 🛑 Human approval for sensitive refund requests  
- 💻 Fully local system (no external APIs required)  

---

## 🏗️ Architecture

```

User Query → Intent Router →

→ Billing Query:
Retrieve (ChromaDB) → LLM → Response

→ Refund Request:
Pause → Human Approval → Resume → Response

→ Out-of-Scope:
Fallback Response

```

The system uses two main flows:
- Automated RAG pipeline  
- Human-in-the-Loop pipeline  

---

## 📊 Source of Truth (Policy Rules)

The assistant strictly follows predefined billing rules:

**Subscription Plans**
- Basic → $10/month  
- Pro → $50/month  
- Enterprise → $200/month  

**Refund Policy**
- ✅ 100% refund within 7 days  
- ❌ No refunds after 14 days  

**Human Approval Required**
- Enterprise plan refunds  
- Refunds greater than $100  

---

## ⚙️ Tech Stack

- LangChain  
- LangGraph  
- ChromaDB  
- Ollama (Phi-3 Mini)  
- Sentence Transformers  
- PyPDF  

---

## 🧱 Project Structure

```

src/
├── rag/
│   ├── ingestion.py
│   └── chain.py
├── graph/
│   ├── router.py
│   └── workflow.py
├── hitl/
│   └── manager_ui.py
└── models.py

data/
├── raw_docs/
├── vectorstore/

docs/
├── HLD.pdf
├── LLD.pdf
├── Technical_Documentation.pdf

````

---

## 🔄 Workflow Explanation

1. User enters a query  
2. Intent is classified  
3. Based on intent:
   - Billing → RAG pipeline  
   - Refund → HITL workflow  
4. Response is generated  

LangGraph manages:
- State transitions  
- Conditional routing  
- Pause & resume functionality  

---

## 🛑 Human-in-the-Loop (HITL)

For sensitive financial actions:

- System pauses execution  
- Manager reviews request  
- Decision: Approve / Reject  
- Workflow resumes  

This ensures **safe AI decision-making**.

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python main.py
````

---

## 💡 Example Queries

* What is the price of the Pro plan?
* Can I get a refund within 5 days?
* Refund my $200 Enterprise plan
* What happens after 3 failed payments?

---

## 🧪 Testing

* ✔ Correct pricing retrieval
* ✔ Refund rule validation
* ✔ HITL trigger for sensitive cases
* ✔ No hallucinated responses

---

## 📄 Documentation

* High-Level Design (HLD)
* Low-Level Design (LLD)
* Technical Documentation


## 🧠 Key Learning

> Not all AI decisions should be automated.

This project demonstrates:

* Safe AI system design
* Real-world LLM application
* Decision-aware workflows

---

## 🚀 Future Improvements

* Multi-document support
* Web-based UI
* User memory integration
* Cloud deployment

--- 
