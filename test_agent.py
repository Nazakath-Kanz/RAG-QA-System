# test_agent.py
import asyncio
from loguru import logger

from src.agent.graph import run_agent


async def main():
    question = "what is the Roll Number?"

    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print('='*60)

    final_state = await run_agent(question)

    print(f"Final query used: {final_state['current_query']}")
    print(f"Retries: {final_state['retry_count']}")
    print(f"Answer: {final_state['answer']}")


if __name__ == "__main__":
    asyncio.run(main())
