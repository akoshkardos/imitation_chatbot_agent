"""Chroma persistence for session documents."""

from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.config import EMBEDDING_MODEL, OPENAI_API_KEY

DEFAULT_PERSIST_DIRECTORY = Path(__file__).resolve().parents[2] / "chroma_db"


def _new_store(persist_directory: str) -> Chroma:
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=OpenAIEmbeddings(
            api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL
        ),
    )


def add_to_vector_store(
    documents, persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY
) -> Chroma:
    """Add only sessions not already present, using deterministic IDs."""
    store = _new_store(persist_directory)
    existing_ids = set(store.get(include=[])["ids"])
    pending, ids = [], []
    for document in documents:
        metadata = document.metadata
        key = "|".join(
            str(value)
            for value in (
                metadata.get("source", ""),
                metadata.get("session_id", ""),
                metadata.get("start_time", ""),
                metadata.get("end_time", ""),
                document.page_content,
            )
        )
        document_id = hashlib.sha256(key.encode("utf-8")).hexdigest()
        if document_id not in existing_ids:
            pending.append(document)
            ids.append(document_id)
    if pending:
        store.add_documents(documents=pending, ids=ids)
    return store


def load_vector_store(
    persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
) -> Chroma:
    """Open a persisted Chroma collection."""
    return _new_store(persist_directory)
