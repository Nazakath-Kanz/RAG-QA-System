# src/generation/llm.py
import os
import google.generativeai as genai
from loguru import logger
from dotenv import load_dotenv


load_dotenv()


GOOGLE_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not GOOGLE_KEY:
    logger.error("System structural alert: 'GOOGLE_API_KEY' is missing from env context.")


if GOOGLE_KEY:
    genai.configure(api_key=GOOGLE_KEY)
else:
    logger.critical("NO VALID GOOGLE API KEY FOUND IN ENVIRONMENT. GENERATION LAYER WILL FAIL.")

async def generate_answer(query: str, contexts: list[str]) -> str:
    """
    Synthesizes highly-precise, context-bounded answers using Google's Gemini LLM engine.
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
    
    try:
        logger.info("Streaming system tokens into the generative model space...")
        

        model = genai.GenerativeModel('gemini-2.5-flash')
        
        full_prompt = f"{system_prompt}\n\nUSER QUERY: {query}"
        
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0
            )
        )
        
        logger.success("Generative inference sequence completed successfully.")
        return response.text.strip()
        
    except Exception as e:
        logger.error(f"Inference boundary fault in Gemini execution payload: {e}")
        
        
        try:
            logger.info("Attempting secondary engine fallback string route...")
            fallback_model = genai.GenerativeModel('gemini-2.5-flash')
            response = fallback_model.generate_content(
                f"{system_prompt}\n\nUSER QUERY: {query}",
                generation_config=genai.types.GenerationConfig(temperature=0.0)
            )
            return response.text.strip()
        except Exception as fallback_err:
            return f"An internal synthesis execution error occurred: {str(fallback_err)}"