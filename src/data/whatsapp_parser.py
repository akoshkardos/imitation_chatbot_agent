"""Parse WhatsApp text exports in the bracketed timestamp format."""

from __future__ import annotations

import re
from pathlib import Path


MESSAGE_PATTERN = re.compile(
    r"^\[(\d{2}-\d{2}-\d{4}, \d{2}:\d{2}:\d{2})\] ([^:]+): (.*)$"
)
SYSTEM_MESSAGE_MARKERS = (
    "end-to-end",
    "berichten en oproepen",
    "messages and calls are end-to-end encrypted",
)
OMITTED_ATTACHMENT_MARKERS = {
    "afbeelding weggelaten": "[Afbeelding]",
    "image omitted": "[Image]",
    "sticker weggelaten": "[Sticker]",
    "sticker omitted": "[Sticker]",
    "video weggelaten": "[Video]",
    "video omitted": "[Video]",
    "audio weggelaten": "[Audio]",
    "audio omitted": "[Audio]",
    "document weggelaten": "[Document]",
    "document omitted": "[Document]",
    "gif weggelaten": "[GIF]",
    "gif omitted": "[GIF]",
    "visitekaartje weggelaten": "[Visitekaartje]",
    "contact card omitted": "[Contact card]",
    "business card omitted": "[Contact card]",
    "<media omitted>": "[Media]",
}


def parse_whatsapp_chat(file_path: str | Path) -> list[dict[str, str]]:
    """Return messages with timestamp, sender and text from a WhatsApp export.

    Lines that do not start a message are appended to the preceding message,
    which handles WhatsApp's multiline messages. System notices are skipped.
    """
    messages: list[dict[str, str]] = []
    current: dict[str, str] | None = None

    def flush() -> None:
        if current is None:
            return
        text = current["text"].strip()
        folded_text = text.casefold()
        if any(marker.casefold() in folded_text for marker in SYSTEM_MESSAGE_MARKERS):
            return
        for marker, replacement in OMITTED_ATTACHMENT_MARKERS.items():
            text = re.sub(re.escape(marker), replacement, text, flags=re.IGNORECASE)
        messages.append({**current, "text": text})

    with Path(file_path).open("r", encoding="utf-8-sig") as chat_file:
        for raw_line in chat_file:
            line = raw_line.rstrip("\r\n")
            match = MESSAGE_PATTERN.match(line)
            if match:
                flush()
                timestamp, sender, first_line = match.groups()
                current = {
                    "timestamp": timestamp,
                    "sender": sender.strip(),
                    "text": first_line,
                }
            elif current is not None:
                current["text"] += "\n" + line

    flush()
    return messages
