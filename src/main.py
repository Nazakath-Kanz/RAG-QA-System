# src/main.py
from src.agent.graph import run_agent
from src.database import qdrant_client
from src.ingestion.pdf_parsing import extract_text_and_tables
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.ingestion.pipeline import ingest_document, init_vector_db
from src.retrieval.search import retrieve_context
from src.generation.llm import generate_answer
from src.models.schemas import QueryRequest, QueryResponse

app = FastAPI(
    title="Enterprise Parent-Child Hierarchical RAG",
    version="2.0.0",
    description="Production-grade asynchronous RAG framework utilizing Qdrant and FlashRank."
)

# Apply CORS middleware to allow cross-origin requests from frontend layers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"],
)

@app.on_event("startup")
async def startup_event():
    """
    Guarantees local database configurations are fully initialized on service startup.
    """
    logger.info("Starting up FastAPI application lifecycle context...")
    await init_vector_db()

@app.post("/upload", status_code=200)
async def upload_document(file: UploadFile = File(...)):
    """
    Ingests, extracts text, and maps documents directly to the Qdrant vector space.
    Accepts .pdf and .txt file streams. Drops old collections for clean data separation.
    """
    logger.info(f"Received file upload stream request for file: '{file.filename}'")
    
    if not file.filename.endswith(('.pdf', '.txt')):
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a .pdf or .txt file.")
        
    try:
        # Read raw byte array stream from the memory buffer
        file_bytes = await file.read()
        extracted_text = ""

        # Handle PDF Text + Table Extraction (structure-aware)
        if file.filename.endswith('.pdf'):
            logger.info(f"Parsing PDF using pdfplumber with table-aware extraction...")
            extracted_text = extract_text_and_tables(file_bytes)

        # Handle Plain Text Extraction
        elif file.filename.endswith('.txt'):
            extracted_text = file_bytes.decode("utf-8")

        # Sanity check on the extracted text length
        if not extracted_text.strip():
            raise HTTPException(status_code=422, detail="Extraction step yielded an empty text body string from document.")

       # --- PURGE EMBEDDING SLATE ---
        # Actually drops the existing collection (init_vector_db alone only creates
        # it if missing — it never deletes stale data from previous uploads).
        logger.info("Purging stale vectors and hard-resetting Qdrant collection space...")
        try:
            await qdrant_client.delete_collection(collection_name="enterprise_parent_child_kb")
            logger.info("Existing collection dropped.")
        except Exception:
            logger.info("No existing collection to drop — proceeding fresh.")
        await init_vector_db()

        # Stream directly into the vector database pipeline using await
        logger.info(f"Awaiting vector matrix graph generation and database upsert execution...")
        await ingest_document(text=extracted_text, filename=file.filename)

        return {
            "status": "Success",
            "detail": f"File '{file.filename}' parsed, vectorized, and safely committed into a clean Qdrant collection."
        }

    except Exception as e:
        logger.error(f"Fatal exception hit during API route text ingestion sequence: {e}")
        raise HTTPException(status_code=500, detail=f"Internal database ingestion stream pipeline error: {str(e)}")

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Advanced context-bounded inference routing.
    Runs Hybrid Querying -> Deep Cross-Encoder Reranking -> Deterministic LLM Generation.
    """
    logger.info(f"Incoming interactive RAG inference query router request: '{request.question}'")
    
    try:
        # Step 1: Query the database and run deep reranking
        context_blocks = await retrieve_context(query=request.question)
        
        # Step 2: Route the bounded retrieved contexts straight to the deterministic LLM
        answer = await generate_answer(query=request.question, contexts=context_blocks)
        
        return QueryResponse(
            query=request.question,
            answer=answer,
            context_used=context_blocks
        )
        
    except Exception as e:
        logger.error(f"Execution failure downstream on the RAG serving inference layer: {e}")
        raise HTTPException(status_code=500, detail=f"RAG Inference Generation Engine pipeline failure: {str(e)}")

@app.post("/ask_agentic", response_model=QueryResponse)
async def ask_question_agentic(request: QueryRequest):
    """
    Agentic inference routing via LangGraph.
    Runs Retrieve -> Generate -> Confidence Check -> (optional) Query Rewrite -> Retry.
    Unlike /ask, this endpoint can self-correct once if the first retrieval
    attempt yields no usable context, by rewriting the query and retrying
    with different phrasing before giving up.
    """
    logger.info(f"Incoming AGENTIC RAG inference query router request: '{request.question}'")

    try:
        final_state = await run_agent(request.question)

        return QueryResponse(
            query=request.question,
            answer=final_state["answer"],
            context_used=final_state["contexts"]
        )

    except Exception as e:
        logger.error(f"Execution failure downstream on the agentic RAG serving layer: {e}")
        raise HTTPException(status_code=500, detail=f"Agentic RAG Inference Engine pipeline failure: {str(e)}")