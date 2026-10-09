"""Small transparent metrics for reviewing answer support."""


def keyword_coverage(answer: str, expected_keywords: list[str]) -> float | None:
    """Fraction of expected terms appearing in an answer; not a truth score."""
    if not expected_keywords:
        return None
    normalized = answer.casefold()
    return sum(term.casefold() in normalized for term in expected_keywords) / len(
        expected_keywords
    )
