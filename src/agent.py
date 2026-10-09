"""Graph-driven WhatsApp conversation simulation."""

import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

from src.config import (
    IMPERSONATED_NAME,
    OPENAI_API_KEY,
    MODEL_NAME,
    TEMPERATURE,
    MAX_TOKENS,
    MAX_RANDOM_CHECKS,
    RANDOM_SESSIONS_PER_CHECK,
)

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

try:
    from src.prompts import SYSTEM_PROMPT
except ModuleNotFoundError as error:
    if error.name != "src.prompts":
        raise
    from src.prompts_example import SYSTEM_PROMPT
from src.tools import (
    retrieve_random_sessions,
    retrieve_relevant_sessions,
    search_sessions_bm25,
)


_EMOJI_BASE = r"[\U0001F000-\U0001FAFF\u2600-\u27BF\u2300-\u23FF\u2B00-\u2BFF\u3030\u303D\u3297\u3299]"
_EMOJI_PATTERN = re.compile(
    rf"(?:[0-9#*]\uFE0F?\u20E3|"
    rf"[\U0001F1E6-\U0001F1FF]{{2}}|"
    rf"{_EMOJI_BASE}(?:\uFE0F|\uFE0E)?"
    rf"(?:[\U0001F3FB-\U0001F3FF])?"
    rf"(?:\u200D{_EMOJI_BASE}(?:\uFE0F)?"
    rf"(?:[\U0001F3FB-\U0001F3FF])?)*|"
    rf"[\U0001F1E6-\U0001F1FF])"
)

llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    max_completion_tokens=MAX_TOKENS,
)


class AgentState(MessagesState, total=False):
    retrieval_query: str
    relevant_context: str
    keyword_context: str
    draft: str
    random_context: str
    random_checks: int
    final_answer: str
    needs_another_random_sample: bool


class RetrievalQuery(BaseModel):
    query: str = Field(
        description=(
            "A natural-language question about the user's latest message, "
            "including useful names or keywords from it."
        )
    )

class BM25Query(BaseModel):
    query: str = Field(
        description=(
            "Up to three useful keywords from the user's latest message for "
            "lexical search."
        )
    )
class ReviewDecision(BaseModel):
    answer: str = Field(description="The revised final answer to the user.")
    needs_another_random_sample: bool = Field(
        description=(
            "True only when another random style sample is likely to improve "
            "the answer; otherwise false."
        )
    )


query_planner = llm.with_structured_output(RetrievalQuery)
answer_reviewer = llm.with_structured_output(ReviewDecision)
bm25_query = llm.with_structured_output(BM25Query)

def latest_user_message(state: AgentState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def plan_retrieval(state: AgentState) -> dict:
    user_text = latest_user_message(state)
    plan = query_planner.invoke(
        [
            SystemMessage(
                content=(
                    "Create a retrieval query for the user's latest message. "
                    "Follow the retrieval tool's usage instructions: "
                    f"{retrieve_relevant_sessions.description} "
                    "Make the query a question and include useful keywords. "
                    "Return only the structured query."
                )
            ),
            HumanMessage(content=user_text),
        ]
    )
    return {
        "retrieval_query": plan.query,
        "relevant_context": "",
        "keyword_context": "",
        "draft": "",
        "random_context": "",
        "random_checks": 0,
        "final_answer": "",
        "needs_another_random_sample": False,
    }


def retrieve_relevant(state: AgentState) -> dict:
    result = retrieve_relevant_sessions.invoke({"query": state["retrieval_query"]})
    return {"relevant_context": str(result)}

def retrieve_bm25_relevant(state: AgentState) -> dict:
    plan = bm25_query.invoke(
        [
            SystemMessage(
                content=(
                    "Extract up to three distinctive keywords or short names "
                    "from the user's latest message for lexical search. "
                    "Return only the structured query."
                )
            ),
            HumanMessage(content=latest_user_message(state)),
        ]
    )
    result = search_sessions_bm25.invoke({"query": plan.query})
    return {"keyword_context": str(result)}

def draft_answer(state: AgentState) -> dict:
    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            *state["messages"],
            SystemMessage(
                content=(
                    "Relevant retrieved conversations:\n"
                    f"{state['relevant_context']}\n\n"
                    "Relevant lexical search results:\n"
                    f"{state['keyword_context']}\n\n"
                    "Draft a concise answer to the latest user message. "
                    "Use this context and the style instructions."
                )
            ),
        ]
    )
    return {"draft": str(response.content)}


def retrieve_random(state: AgentState) -> dict:
    result = retrieve_random_sessions.invoke(
        {"number": RANDOM_SESSIONS_PER_CHECK}
    )
    accumulated = state.get("random_context", "")
    check_number = state.get("random_checks", 0) + 1
    return {
        "random_context": f"{accumulated}\n\n{result}".strip(),
        "random_checks": check_number,
    }


def review_answer(state: AgentState) -> dict:
    decision = answer_reviewer.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            *state["messages"],
            SystemMessage(
                content=(
                    "Review and improve the draft using the relevant sessions "
                    "and the random sessions as style examples. Match "
                    f"{IMPERSONATED_NAME}'s "
                    "tone, wording, message length, and usual sentence pattern. "
                    "Preserve the casing, spelling, and punctuation of the "
                    "closest style examples; do not normalize casual lowercase "
                    "or add periods and commas automatically. Keep the final "
                    "message to one short line, at most 12 words unless the user "
                    "asks for detail. Return only the message, without an "
                    "explanation or preamble. "
                    "Follow the emoji limit in the system instructions exactly. "
                    "Keep the answer relevant to the user's latest message. "
                    "Decide whether one more random sample is likely to help. "
                    "If you request one, "
                    "still provide your best current answer.\n\n"
                    f"Relevant sessions:\n{state['relevant_context']}\n\n"
                    f"Relevant keyword based sessions:\n{state['keyword_context']}\n\n"
                    f"Draft answer:\n{state['draft']}\n\n"
                    f"Random style samples so far:\n{state['random_context']}\n\n"
                    f"Random checks already run: {state['random_checks']} "
                    f"(maximum {MAX_RANDOM_CHECKS})."
                )
            ),
        ]
    )
    return {
        "final_answer": decision.answer,
        "needs_another_random_sample": decision.needs_another_random_sample,
    }


def route_after_review(state: AgentState) -> str:
    if (
        state.get("needs_another_random_sample", False)
        and state.get("random_checks", 0) < MAX_RANDOM_CHECKS
    ):
        return "retrieve_random"
    return "finish"


def enforce_emoji_limit(answer: str, state: AgentState) -> str:
    """Allow at most one emoji in any six consecutive assistant replies."""
    prior_answers = [
        str(message.content)
        for message in state["messages"]
        if isinstance(message, AIMessage)
    ][-5:]
    if any(_EMOJI_PATTERN.search(previous) for previous in prior_answers):
        return _EMOJI_PATTERN.sub("", answer)

    found = list(_EMOJI_PATTERN.finditer(answer))
    if len(found) <= 1:
        return answer

    first_emoji_end = found[0].end()
    return answer[:first_emoji_end] + _EMOJI_PATTERN.sub("", answer[first_emoji_end:])


def finish(state: AgentState) -> dict:
    answer = enforce_emoji_limit(state["final_answer"], state)
    return {"messages": [AIMessage(content=answer)]}


workflow = StateGraph(AgentState)
workflow.add_node("plan_retrieval", plan_retrieval)
workflow.add_node("retrieve_relevant", retrieve_relevant)
workflow.add_node("retrieve_bm25_relevant", retrieve_bm25_relevant)
workflow.add_node("draft_answer", draft_answer)
workflow.add_node("retrieve_random", retrieve_random)
workflow.add_node("review_answer", review_answer)
workflow.add_node("finish", finish)
workflow.add_edge(START, "plan_retrieval")
workflow.add_edge("plan_retrieval", "retrieve_relevant")
workflow.add_edge("retrieve_relevant", "retrieve_bm25_relevant")
workflow.add_edge("retrieve_bm25_relevant", "draft_answer")
workflow.add_edge("draft_answer", "retrieve_random")
workflow.add_edge("retrieve_random", "review_answer")
workflow.add_conditional_edges(
    "review_answer",
    route_after_review,
    {"retrieve_random": "retrieve_random", "finish": "finish"},
)
workflow.add_edge("finish", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "test"}}

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        result = app.invoke(
            {"messages": [HumanMessage(content=user_input)]}, config=config
        )
        print(f"{IMPERSONATED_NAME}:", result["messages"][-1].content)
