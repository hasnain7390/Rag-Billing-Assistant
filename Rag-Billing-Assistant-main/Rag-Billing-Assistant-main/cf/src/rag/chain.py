from __future__ import annotations

from pathlib import Path

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _build_embeddings() -> HuggingFaceEmbeddings:
    # Keep embedding stack aligned with ingestion/indexing.
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda" if _cuda_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def build_retriever(k: int = 3):
    persist_dir = _project_root() / "data" / "vectorstore"
    vectorstore = Chroma(
        collection_name="billing_policy",
        embedding_function=_build_embeddings(),
        persist_directory=str(persist_dir),
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})


def build_rag_chain(model_name: str = "phi3:mini", k: int = 3):
    llm = ChatOllama(model=model_name, temperature=0)
    retriever = build_retriever(k=k)

    prompt = ChatPromptTemplate.from_template(
        """
You are a professional SaaS Billing Assistant.
Answer the user's question using ONLY the provided context.
If the answer is not in the context, politely state that you do not have that information and suggest contacting manager@saas.com.

Context:
{context}

Question:
{input}
""".strip()
    )

    qa_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    return create_retrieval_chain(retriever=retriever, combine_docs_chain=qa_chain)


def run_query(chain, question: str) -> None:
    result = chain.invoke({"input": question})

    print("\n" + "=" * 80)
    print(f"Question: {question}")
    print("Answer:")
    print(result.get("answer", "<no answer>"))

    print("\nSource metadata:")
    for idx, doc in enumerate(result.get("context", []), start=1):
        metadata = doc.metadata or {}
        source = metadata.get("source", "unknown")
        page = metadata.get("page", "unknown")
        snippet = doc.page_content.strip().replace("\n", " ")[:180]
        print(f"{idx}. source={source}, page={page}, snippet={snippet}")


if __name__ == "__main__":
    rag_chain = build_rag_chain(model_name="phi3:mini", k=3)

    run_query(rag_chain, "What are the rules for Enterprise plan refunds?")
    run_query(rag_chain, "How many failed payments lead to account suspension?")
