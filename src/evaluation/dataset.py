"""Load hand-authored evaluation cases from JSON Lines."""

import json
from pathlib import Path


def load_cases(path: Path) -> list[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as data_file:
        for line_number, line in enumerate(data_file, start=1):
            if not line.strip():
                continue
            case = json.loads(line)
            if not isinstance(case.get("question"), str) or not case["question"].strip():
                raise ValueError(f"Line {line_number}: question must be non-empty text")
            expected_answer = case.get("expected_answer", "")
            if not isinstance(expected_answer, str):
                raise ValueError(f"Line {line_number}: expected_answer must be text")
            expected_keywords = case.get("expected_keywords", [])
            if not isinstance(expected_keywords, list) or not all(
                isinstance(keyword, str) for keyword in expected_keywords
            ):
                raise ValueError(
                    f"Line {line_number}: expected_keywords must be a list of strings"
                )
            case["expected_answer"] = expected_answer
            case.setdefault("expected_keywords", [])
            cases.append(case)
    return cases
