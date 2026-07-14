"""
LangGraph agentic layer wrapping the existing retrieval + generation pipeline.

Adds one capability the base pipeline does not have: if the first retrieval
attempt fails (falls back to "no relevant context"), the agent rewrites the
query with different phrasing and retries once before giving up. This
directly targets the phrasing-sensitivity issue found during manual testing
(e.g. "roll no" succeeding while "roll number" failed on the same data).
"""

from typing_extensions import TypedDict
from loguru import logger
import google.generativeai as genai

from langgraph.graph import StateGraph, START, END

from src.retrieval.search import retrieve_context
from src.generation.llm import generate_answer

FALLBACK_MESSAGE = "No relevant context found in the database to answer this question securely."
MAX_RETRIES = 1


class AgentState(TypedDict):
    original_question: str
    current_query: str
    contexts: list[str]
    answer: str
    retry_count: int


async def retrieve_node(state: AgentState) -> dict:
    logger.info(f"[retrieve] query: '{state['current_query']}'")
    contexts = await retrieve_context(state["current_query"])
    return {"contexts": contexts}


async def generate_node(state: AgentState) -> dict:
    logger.info(f"[generate] generating answer from {len(state['contexts'])} context(s)")
    answer = await generate_answer(query=state["original_question"], contexts=state["contexts"])
    return {"answer": answer}


async def rewrite_query_node(state: AgentState) -> dict:
    logger.info(f"[rewrite_query] rephrasing failed query: '{state['current_query']}'")
    prompt = (
        "Rewrite the following question using different wording or synonyms, "
        "keeping the same meaning, so it might match different phrasing in a document. "
        "Return ONLY the rewritten question, nothing else.\n\n"
        f"Question: {state['current_query']}"
    )
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    rewritten = response.text.strip()
    logger.info(f"[rewrite_query] rewritten to: '{rewritten}'")
    return {
        "current_query": rewritten,
        "retry_count": state["retry_count"] + 1,
    }


def check_confidence(state: AgentState) -> str:
    if state["answer"].strip() == FALLBACK_MESSAGE and state["retry_count"] < MAX_RETRIES:
        logger.info("[check_confidence] low confidence, retrying with rewritten query")
        return "rewrite_query"
    logger.info("[check_confidence] accepting answer")
    return "end"


builder = StateGraph(AgentState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_node)
builder.add_node("rewrite_query", rewrite_query_node)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_conditional_edges(
    "generate",
    check_confidence,
    {"rewrite_query": "rewrite_query", "end": END},
)
builder.add_edge("rewrite_query", "retrieve")

agent_graph = builder.compile()


async def run_agent(question: str) -> dict:
    """Runs the full agentic graph for a single question and returns the final state."""
    initial_state: AgentState = {
        "original_question": question,
        "current_query": question,
        "contexts": [],
        "answer": "",
        "retry_count": 0,
    }
    final_state = await agent_graph.ainvoke(initial_state)
    return final_state
