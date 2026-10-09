import json
import random

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.tools import tool

from src.indexing.vector_store import DEFAULT_PERSIST_DIRECTORY, load_vector_store

vector_store = load_vector_store()
bm25_corpus_path = DEFAULT_PERSIST_DIRECTORY / "bm25_documents.json"
if bm25_corpus_path.exists():
    with bm25_corpus_path.open(encoding="utf-8") as corpus_file:
        bm25_items = json.load(corpus_file)
    bm25_documents = [
        Document(page_content=item["page_content"], metadata=item["metadata"])
        for item in bm25_items
    ]
    bm25_retriever = (
        BM25Retriever.from_documents(bm25_documents, k=5)
        if bm25_documents
        else None
    )
else:
    bm25_retriever = None


@tool
def search_sessions_bm25(query: str) -> str:
    """Search conversation sessions by matching query words lexically."""
    if bm25_retriever is None:
        return "BM25 corpus not found. Build the Chroma index first."

    docs = bm25_retriever.invoke(query)
    if not docs:
        return "No matching conversations found."

    return "Lexical search results:\n\n" + "\n\n".join(
        f"--- Conversation {index} ---\n{doc.page_content}\n"
        f"(Date and time: {doc.metadata['start_time']})"
        for index, doc in enumerate(docs, start=1)
    )


@tool
def retrieve_relevant_sessions(query):
    """
    Use this tool to search for conversation sessions 
    that are relevant to the thing said by the user.
    Always query a question and implement useful keywords, so you
    can find relevant sessions, word only calls don't work well.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )
    docs = retriever.invoke(query)

    if not docs:
        return "Geen relevante gesprekken gevonden."
    
    context = "Relevant conversations:\n\n"
    for i, doc in enumerate(docs):
        source = doc.metadata["source"]
        session_id = doc.metadata["session_id"]

        context += f"--- Conversation {i+1} ---\n"
        context += f"{doc.page_content}\n"
        context += f"(Date and time this occured: {doc.metadata['start_time']})\n\n"

        previous = vector_store.get(
            where={
                "$and": [
                    {"source": source},
                    {"session_id": session_id - 1}
                ]
            }
        )

        if previous["ids"]:
            context += "--- Previous Conversation ---\n"
            context += f"{previous['documents'][0]}\n"
            context += (
                f"(Source: {source})\n"
                f"(Date and time: "
                f"{previous['metadatas'][0]['start_time']})\n\n"
            )

        # Next session
        next_session = vector_store.get(
            where={
                "$and": [
                    {"source": source},
                    {"session_id": session_id + 1}
                ]
            }
        )

        if next_session["ids"]:
            context += "--- Next Conversation ---\n"
            context += f"{next_session['documents'][0]}\n"
            context += (
                f"(Source: {source})\n"
                f"(Date and time: "
                f"{next_session['metadatas'][0]['start_time']})\n\n"
            )
    return context


@tool
def retrieve_random_sessions(number: int):
    """Retrieve a number of randomly selected documents from the vector store."""
    if number <= 0:
        return "Positive integers only"

    stored = vector_store.get(include=[])
    ids = stored["ids"]

    if not ids:
        return "No documents found."

    selected_ids = random.sample(
        ids,
        min(number, len(ids))
    )

    selected = vector_store.get(
        ids=selected_ids,
        include=["documents", "metadatas"]
    )

    context = "Random conversations:\n\n"

    for index, (document, metadata) in enumerate(
        zip(selected["documents"], selected["metadatas"]),
        start=1
    ):
        context += f"--- Random Conversation {index} ---\n"
        context += f"{document}\n"
        context += f"(Source: {metadata['source']})\n"
        context += f"(Date and time: {metadata['start_time']})\n\n"

    return context
