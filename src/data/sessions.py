"""Group messages into sessions and convert them to vector-store documents."""

from __future__ import annotations

from datetime import datetime, timedelta

from langchain_core.documents import Document

from src.config import SESSION_GAP_MINUTES


TIMESTAMP_FORMAT = "%d-%m-%Y, %H:%M:%S"


def group_into_sessions(
    messages: list[dict[str, str]], gap_minutes: int = SESSION_GAP_MINUTES
) -> list[list[dict[str, object]]]:
    """Sort messages by timestamp and group gaps over ``gap_minutes``."""
    if gap_minutes < 0:
        raise ValueError("gap_minutes must be non-negative")

    # Exports can contain messages out of order (for example, after a merge).
    # Stable sorting preserves the original order when timestamps are equal.
    ordered_messages = sorted(
        messages,
        key=lambda message: datetime.strptime(
            str(message["timestamp"]), TIMESTAMP_FORMAT
        ),
    )

    sessions: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    previous_timestamp: datetime | None = None

    for message in ordered_messages:
        timestamp = datetime.strptime(str(message["timestamp"]), TIMESTAMP_FORMAT)
        if current and previous_timestamp is not None:
            if timestamp - previous_timestamp > timedelta(minutes=gap_minutes):
                sessions.append(current)
                current = []
        current.append({**message, "ts": timestamp})
        previous_timestamp = timestamp

    if current:
        sessions.append(current)
    return sessions


def sessions_to_documents(
    sessions: list[list[dict[str, object]]], source: str
) -> list[Document]:
    """Convert sessions to Chroma-compatible LangChain documents."""
    documents: list[Document] = []
    for session_id, session in enumerate(sessions, start=1):
        if not session:
            continue
        lines = [f"{message['sender']}: {message['text']}" for message in session]
        participants = sorted({str(message["sender"]) for message in session})
        start = session[0]["ts"]
        end = session[-1]["ts"]
        if not isinstance(start, datetime) or not isinstance(end, datetime):
            raise ValueError("Session messages must be grouped before conversion")
        documents.append(
            Document(
                page_content="\n".join(lines),
                metadata={
                    "source": source,
                    "session_id": session_id,
                    "start_time": start.isoformat(),
                    "end_time": end.isoformat(),
                    "participants": ", ".join(participants),
                    "message_count": len(session),
                },
            )
        )
    return documents
