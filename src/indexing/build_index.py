"""Build or update the local Chroma index from a WhatsApp export."""

import json
from pathlib import Path

from src.data import group_into_sessions, parse_whatsapp_chat, sessions_to_documents
from src.config import SESSION_GAP_MINUTES
from src.indexing.vector_store import add_to_vector_store


def build_index(
    chat_file: Path,
    persist_directory: Path,
    gap_minutes: int = SESSION_GAP_MINUTES,
):
    messages = parse_whatsapp_chat(chat_file)
    sessions = group_into_sessions(messages, gap_minutes=gap_minutes)
    documents = sessions_to_documents(sessions, source=chat_file.stem)
    store = add_to_vector_store(documents, persist_directory=str(persist_directory))

    # Export the complete collection so the BM25 corpus also includes documents
    # that were already indexed before this build.
    collection = store.get(include=["documents", "metadatas"])
    bm25_documents = [
        {"page_content": content, "metadata": metadata}
        for content, metadata in zip(
            collection["documents"], collection["metadatas"]
        )
    ]
    persist_directory = Path(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)
    (persist_directory / "bm25_documents.json").write_text(
        json.dumps(bm25_documents, ensure_ascii=False), encoding="utf-8"
    )

    return store
