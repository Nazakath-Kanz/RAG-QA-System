# src/main.py
import pypdf
from io import BytesIO
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

        # Handle PDF Text Extraction
        if file.filename.endswith('.pdf'):
            logger.info(f"Parsing binary PDF streams using pypdf reader extractor...")
            pdf_reader = pypdf.PdfReader(BytesIO(file_bytes))
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
                    
        # Handle Plain Text Extraction
        elif file.filename.endswith('.txt'):
            extracted_text = file_bytes.decode("utf-8")

        # Sanity check on the extracted text length
        if not extracted_text.strip():
            raise HTTPException(status_code=422, detail="Extraction step yielded an empty text body string from document.")

        # --- PURGE EMBEDDING SLATE ---
        # Forces Qdrant to drop its current collection and instantiate a clean index structure.
        # This prevents overlapping chunk vectors from bleeding into comparative queries.
        logger.info("Purging stale vectors and hard-resetting Qdrant collection space...")
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