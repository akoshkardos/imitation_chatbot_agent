"""Inspect parsed WhatsApp data without printing message contents."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from src.data import group_into_sessions, parse_whatsapp_chat


def inspect(path: Path, gap_minutes: int = 180) -> None:
    messages = parse_whatsapp_chat(path)
    sessions = group_into_sessions(messages, gap_minutes=gap_minutes)
    senders = Counter(message["sender"] for message in messages)
    sizes = [len(session) for session in sessions]

    print(f"File: {path}")
    print(f"Messages parsed: {len(messages)}")
    print(f"Distinct senders: {len(senders)}")
    sender_labels = {sender: f"Sender {index}" for index, sender in enumerate(senders, 1)}
    for sender, count in senders.items():
        print(f"  {sender_labels[sender]}: {count}")
    print(f"Sessions (gap > {gap_minutes} min): {len(sessions)}")
    if sizes:
        print(
            "Messages per session: "
            f"min={min(sizes)}, median={sorted(sizes)[len(sizes) // 2]}, "
            f"max={max(sizes)}"
        )
    if not messages:
        print("No messages matched the expected export format.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chat_file", type=Path, help="WhatsApp .txt export")
    parser.add_argument("--gap-minutes", type=int, default=180)
    args = parser.parse_args()
    inspect(args.chat_file, args.gap_minutes)


if __name__ == "__main__":
    main()
