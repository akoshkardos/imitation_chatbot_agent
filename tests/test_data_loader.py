"""Unit tests for WhatsApp parsing and session grouping."""

import tempfile
import unittest
from pathlib import Path

from src.data import group_into_sessions, parse_whatsapp_chat, sessions_to_documents


class WhatsAppParserTests(unittest.TestCase):
    def parse_text(self, text: str):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chat.txt"
            path.write_text(text, encoding="utf-8")
            return parse_whatsapp_chat(path)

    def test_parses_messages_and_multiline_text(self):
        messages = self.parse_text(
            "[01-02-2024, 10:00:00] Noor: first line\ncontinued\n"
            "[01-02-2024, 10:01:00] Sam: hello: there\n"
        )
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["sender"], "Noor")
        self.assertEqual(messages[0]["text"], "first line\ncontinued")
        self.assertEqual(messages[1]["text"], "hello: there")

    def test_skips_system_notices_and_normalizes_image_placeholder(self):
        messages = self.parse_text(
            "[01-02-2024, 10:00:00] Noor: afbeelding weggelaten\n"
            "[01-02-2024, 10:01:00] System: end-to-end encrypted\n"
        )
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["text"], "[Afbeelding]")

    def test_groups_by_gap_and_converts_to_documents(self):
        messages = self.parse_text(
            "[01-02-2024, 10:00:00] Noor: hello\n"
            "[01-02-2024, 10:02:00] Sam: hi\n"
            "[01-02-2024, 14:00:00] Noor: later\n"
        )
        sessions = group_into_sessions(messages, gap_minutes=180)
        self.assertEqual([len(session) for session in sessions], [2, 1])
        docs = sessions_to_documents(sessions, source="test")
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].metadata["message_count"], 2)
        self.assertIn("Noor: hello", docs[0].page_content)

    def test_sorts_out_of_order_messages_before_grouping(self):
        messages = self.parse_text(
            "[01-02-2024, 10:02:00] Noor: later\n"
            "[01-02-2024, 10:00:00] Sam: earlier\n"
        )
        sessions = group_into_sessions(messages)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0][0]["sender"], "Sam")
        self.assertEqual(sessions[0][1]["sender"], "Noor")


if __name__ == "__main__":
    unittest.main()
