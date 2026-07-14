import asyncio
from src.retrieval.search import retrieve_context

async def main():
    questions = [
        "what is the subject code for science?",
        "what are the theory marks for science?",
        "what is the roll number?",
        "what is the total marks for malayalam?",
    ]

    for question in questions:
        print(f"\n{'='*60}")
        print(f"QUESTION: {question}")
        print('='*60)
        contexts = await retrieve_context(question)
        if not contexts:
            print("No context retrieved.")
        else:
            for i, ctx in enumerate(contexts, start=1):
                print(f"\n--- Context {i} ---")
                print(ctx)

if __name__ == "__main__":
    asyncio.run(main())