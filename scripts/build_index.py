"""Command line entry point for indexing a WhatsApp export."""

import argparse
from pathlib import Path

from src.config import SESSION_GAP_MINUTES
from src.indexing.build_index import build_index
from src.indexing.vector_store import DEFAULT_PERSIST_DIRECTORY


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chat_file", type=Path)
    parser.add_argument("--store", type=Path, default=DEFAULT_PERSIST_DIRECTORY)
    parser.add_argument("--gap-minutes", type=int, default=SESSION_GAP_MINUTES)
    args = parser.parse_args()
    store = build_index(args.chat_file, args.store, args.gap_minutes)
    print(f"Index ready ({store._collection.count()} sessions).")


if __name__ == "__main__":
    main()
