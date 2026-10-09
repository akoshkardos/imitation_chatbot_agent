"""Run fixed questions through the chat graph and save reviewable results."""

import json
from pathlib import Path

from langchain_core.messages import HumanMessage

from src.evaluation.dataset import load_cases
from src.evaluation.metrics import keyword_coverage


def run(cases_path: Path, output_path: Path) -> None:
    # Import only when evaluation is requested: the agent also initializes API tools.
    from src.agent import app

    cases = load_cases(cases_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for index, case in enumerate(cases, start=1):
            config = {"configurable": {"thread_id": f"eval-{index}"}}
            result = app.invoke({"messages": [HumanMessage(content=case["question"])]}, config)
            messages = result["messages"]
            answer_content = messages[-1].content
            answer = (
                answer_content
                if isinstance(answer_content, str)
                else json.dumps(answer_content, ensure_ascii=False)
            )
            score = keyword_coverage(answer, case["expected_keywords"])
            record = {
                "case": index,
                "question": case["question"],
                "expected_answer": case["expected_answer"],
                "agent_answer": answer,
                "expected_keywords": case["expected_keywords"],
                "score": score,
                "score_type": "keyword_coverage",
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"\n{'=' * 72}\nCase {index}/{len(cases)}")
            print(f"Question: {record['question']}")
            print(f"Expected answer: {record['expected_answer'] or '[not provided]'}")
            print(f"Agent answer: {record['agent_answer']}")
            if score is None:
                print("Score (keyword coverage): N/A - add expected_keywords for this case")
            else:
                print(f"Score (keyword coverage): {score:.2f} ({score:.0%})")
        print(f"\nSaved all {len(cases)} case results to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    local_cases = Path("evals/questions.jsonl")
    example_cases = Path("evals/questions.example.jsonl")
    default_cases = local_cases if local_cases.exists() else example_cases
    parser.add_argument("--cases", type=Path, default=default_cases)
    parser.add_argument("--output", type=Path, default=Path("evals/results/latest.jsonl"))
    args = parser.parse_args()
    run(args.cases, args.output)
