# src/generation/llm.py
from langsmith import traceable
import os
from loguru import logger
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GOOGLE_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not GOOGLE_KEY:
    logger.error("System structural alert: 'GOOGLE_API_KEY' is missing from env context.")
    logger.critical("NO VALID GOOGLE API KEY FOUND IN ENVIRONMENT. GENERATION LAYER WILL FAIL.")

# New Google Gen AI SDK client (replaces the deprecated google.generativeai package).
# Using .aio for async support, since generate_answer is called with await elsewhere.
client = genai.Client(api_key=GOOGLE_KEY)
@traceable(name="generate_answer", tags=["generation"])
async def generate_answer(query: str, contexts: list[str]) -> str:
    """
    Synthesizes highly-precise, context-bounded answers using Google's Gemini LLM engine
    via the current google-genai SDK.
    """
    if not contexts:
        logger.warning("Generation layer received empty context array. Defaulting to defensive fallback response.")
        return "No relevant context found in the database to answer this question securely."

    logger.info("Assembling context matrix structures for token processing...")

    context_str = "\n\n---\n\n".join(contexts)

    system_prompt = (
        "You are an elite, objective technical recruiting coordinator assistant. Your goal is to synthesize "
        "accurate answers using ONLY the provided verified document contexts below.\n"
        "STRICT INSTRUCTIONS:\n"
        "1. Answer the query truthfully based solely on the text provided.\n"
        "2. If the context does not explicitly contain the details to formulate an answer, state: "
        "'No relevant context found in the database to answer this question securely.'\n"
        "3. Do NOT hallucinate, guess, or utilize external pre-trained model biases.\n\n"
        f"VERIFIED CONTEXTS:\n{context_str}"
    )

    full_prompt = f"{system_prompt}\n\nUSER QUERY: {query}"

    try:
        logger.info("Streaming system tokens into the generative model space...")

        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0.0
            ),
        )

        logger.success("Generative inference sequence completed successfully.")
        return response.text.strip()

    except Exception as e:
        logger.error(f"Inference boundary fault in Gemini execution payload: {e}")
        return f"An internal synthesis execution error occurred: {str(e)}"