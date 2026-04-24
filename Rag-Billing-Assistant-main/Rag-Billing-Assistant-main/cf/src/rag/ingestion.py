from __future__ import annotations

import os
import shutil
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _paths() -> tuple[Path, Path]:
    root = _project_root()
    pdf_path = root / "data" / "raw_docs" / "billing_policy.pdf"
    persist_dir = root / "data" / "vectorstore"
    return pdf_path, persist_dir


def load_and_split_policy(pdf_path: Path):
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    # 500/50 keeps chunks small enough for local models (better retrieval precision)
    # while preserving nearby context across chunk boundaries.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def build_embeddings() -> HuggingFaceEmbeddings:
    # LangChain will use CUDA when torch reports it is available.
    device = "cuda" if _cuda_available() else "cpu"
    print(f"Embedding device: {device}")

    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def get_or_create_vectorstore(reindex: bool = False) -> Chroma:
    pdf_path, persist_dir = _paths()

    if not pdf_path.exists():
        raise FileNotFoundError(f"Policy PDF not found at: {pdf_path}")

    if reindex and persist_dir.exists():
        shutil.rmtree(persist_dir)

    persist_dir.mkdir(parents=True, exist_ok=True)

    embeddings = build_embeddings()
    vectorstore = Chroma(
        collection_name="billing_policy",
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    # Avoid duplicate inserts on reruns when existing persisted vectors are present.
    existing_count = vectorstore._collection.count()
    if existing_count == 0 or reindex:
        chunks = load_and_split_policy(pdf_path)
        vectorstore.add_documents(chunks)
        try:
            vectorstore.persist()
        except Exception:
            # Newer Chroma versions persist automatically; explicit persist may be a no-op.
            pass
        print(f"Indexed {len(chunks)} chunks into {persist_dir}")
    else:
        print(f"Using existing persisted index with {existing_count} chunks at {persist_dir}")

    return vectorstore


def smoke_test(vectorstore: Chroma) -> None:
    query = "How much is the Enterprise plan?"
    results = vectorstore.similarity_search(query, k=3)

    print("\nSmoke test query:", query)
    print("Top matches:")
    for i, doc in enumerate(results, start=1):
        text = doc.page_content.strip().replace("\n", " ")
        print(f"{i}. {text[:220]}")


if __name__ == "__main__":
    # Set REINDEX=true to force rebuilding from PDF.
    reindex_flag = os.getenv("REINDEX", "false").lower() == "true"
    store = get_or_create_vectorstore(reindex=reindex_flag)
    smoke_test(store)
