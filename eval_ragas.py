# eval_ragas.py
"""
Ragas evaluation harness - expanded golden set (8 questions).
Runs each through the actual retrieval + generation pipeline, then scores
Faithfulness using Gemini as judge. Retries capped to avoid quota loops.
"""

import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.metrics import Faithfulness
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from langchain_google_genai import ChatGoogleGenerativeAI

from src.retrieval.search import retrieve_context
from src.generation.llm import generate_answer

load_dotenv()

GOLDEN_SET = [
    {
        "question": "what is the subject code for science?",
        "reference": "The subject code for Science is 086.",
    },
    {
        "question": "what is the roll number?",
        "reference": "The roll number is 24162608.",
    },
    {
        "question": "what is the total marks for malayalam?",
        "reference": "The total marks for Malayalam is 084 (Eighty Four).",
    },
    {
        "question": "what is Kanz's date of birth?",
        "reference": "Kanz's date of birth is 16/06/2004 (16th June 2004).",
    },
    {
        "question": "what grade did Kanz get in mathematics?",
        "reference": "Kanz got grade B2 in Mathematics Standard.",
    },
    {
        "question": "who is Kanz's mother?",
        "reference": "Kanz's mother's name is Thahmeena K.",
    },
    {
        "question": "what school did Kanz attend?",
        "reference": "Kanz attended MES Central School, Puthiyangadi, Kozhikode, Kerala.",
    },
    {
        "question": "what is Kanz's Aadhaar number?",
        "reference": "This information is not present in the document, so the system should state no relevant context was found.",
    },
]

EVAL_RUN_CONFIG = RunConfig(
    max_retries=2,
    max_wait=15,
    timeout=60,
)


async def build_samples() -> list[SingleTurnSample]:
    samples = []

    for item in GOLDEN_SET:
        question = item["question"]
        reference = item["reference"]

        logger.info(f"Running pipeline for: '{question}'")

        contexts = await retrieve_context(question)
        answer = await generate_answer(query=question, contexts=contexts)

        logger.info(f"  Retrieved {len(contexts)} context(s)")
        logger.info(f"  Answer: {answer}")

        samples.append(
            SingleTurnSample(
                user_input=question,
                retrieved_contexts=contexts if contexts else [""],
                response=answer,
                reference=reference,
            )
        )

    return samples


async def main():
    samples = await build_samples()
    dataset = EvaluationDataset(samples=samples)

    google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    judge_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
        google_api_key=google_api_key,
    )
    evaluator_llm = LangchainLLMWrapper(judge_llm)

    logger.info("Running Ragas evaluation (Faithfulness only, capped retries)...")

    try:
        result = evaluate(
            dataset=dataset,
            metrics=[Faithfulness(llm=evaluator_llm)],
            run_config=EVAL_RUN_CONFIG,
        )
    except Exception as e:
        logger.error(f"Evaluation failed, likely due to quota exhaustion: {e}")
        logger.error("Check https://ai.dev/rate-limit for current usage, and retry after quota resets.")
        return

    print("\n" + "=" * 60)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 60)
    print(result)

    df = result.to_pandas()
    df.to_csv("ragas_eval_results.csv", index=False)
    logger.success("Results saved to ragas_eval_results.csv")


if __name__ == "__main__":
    asyncio.run(main())
