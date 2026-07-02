# Enterprise Parent-Child Hierarchical RAG Platform v2.0.0

A production-grade, asynchronous Retrieval-Augmented Generation (RAG) framework engineered to parse, structurally isolate, and dynamically query dense technical data profiles (such as corporate resumes, developer profiles, or legal dossiers) under strict semantic boundaries. Built natively using FastAPI (Async ASGI), Qdrant High-Performance Vector Database, FlashRank Cross-Encoder Neural Reranking, and the Google Gemini Flash API.

---

## 1. System Architecture & Processing Mechanics

This platform shifts entirely away from naive, superficial chunking models (e.g., standard flat character splitters), implementing an enterprise-tier decoupled retrieval-generation topology explicitly designed to maximize mathematical search precision and completely eliminate LLM hallucinations.

```
                  ┌────────────────────────────────────────┐
                  │          User Input Query              │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │    Qdrant Dense Vector Space Scan      │
                  │  (Fetches Top 15 Granular Child Chunks)│
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │    FlashRank Neural Cross-Encoder      │
                  │ (Local ms-marco Model Filters Top 6)   │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │      Deterministic LLM Engine          │
                  │ (Temperature=0.0 + Full Parent Context)│
                  └────────────────────────────────────────┘
```

### Key Architectural Pillars:
* **Parent-Child Chunking Strategy:** Navigates the fundamental trade-off of vector math: short text strings cluster with hyper-precision but lack comprehensive context; large text blocks retain context but dilute vector distances. This design embeds fine-grained Child Nodes for mathematical indexing, but resolves queries using the complete bounding Parent Document paragraphs.
* **Neural Cross-Encoder Reranking:** Bi-encoder models (standard vector searches) are excellent for high-speed initial retrieval but struggle with deep semantic interplay. By routing candidate vectors into a local ms-marco-MiniLM-L-12-v2 Cross-Encoder, the pipeline scores exact query-document compatibility before routing data to the cloud API, saving massive token costs.
* **Deterministic Inference Firewall:** Enforces temperature=0.0 paired with strict context system boundaries. If the injected vector context blocks do not explicitly provide concrete answers, the LLM safely defaults to an active fallback state rather than generating speculative hallucinations.

---

##  2. Core Technical Stack

* **Backend Routing Core:** FastAPI (Asynchronous Python ASGI runtime)
* **Vector Database:** Qdrant Engine (Asynchronous cluster running local memory/disk maps)
* **Neural Rerank Optimization:** FlashRank Engine (Utilizing the ms-marco-MiniLM-L-12-v2 Cross-Encoder model)
* **Vector Text Embeddings:** SentenceTransformers (all-MiniLM-L6-v2 producing 384-dimensional spatial maps)
* **Generative Inference:** Google Gemini Pro / Flash AI API
* **Interactive Frontend:** Streamlit Dashboard (State-managed dual-pane telemetry layout)

---

##  3. Verified System Test Suites & Assertions

###  Successful Assertions (Passed Production Verification)
1. **Granular Value Lookup (Direct Structural Hit)**
   * Query: "What did Kanz build for House of Courton?"
   * System Output: Successfully extracted precise technical specs (LSTM, GRU, Transformers), target metrics (MAE: 0.015, RMSE: 0.019), and operational pricing ranges ($0.05–0.25/kWh) from deep within a specific data sub-block.
2. **Context-Jumping Verification (Metadata Synthesis)**
   * Query: "What are the details regarding Kanz's B.Sc. education?"
   * System Output: Automatically bypassed historical work history blocks to cleanly isolate Yenepoya Institute parameters, the 8.1/10 GPA baseline, and core foundational AI/ML coursework matrices.
3. **Inventory Isolation (Skills Matrix Extraction)**
   * Query: "What machine learning libraries and deployment tools does Kanz have verified experience using?"
   * System Output: Consolidated inventory skills lists, returning explicit tags (PyTorch, TensorFlow, FastAPI, Docker, Gradio) without introducing external domain noise.

###  Observed Edge Cases & Post-Mortem Analysis
* **Test Scenario:** Multi-Variable Aggregated Comparative Query ("Compare latency, concurrency, and streaming across IBM, Banao Tech, and primetrade.ai").
* **Observed System Behavior:** The pipeline output flawless aggregated data points for Banao Tech and primetrade.ai, but fell back safely to an anti-hallucination neutral response ("No relevant context found in the database to answer this question securely") for the IBM project parameters.
* **Root-Cause Analysis (Semantic Vector Displacement):** When a user query combines three heavily detailed technical projects simultaneously, the mathematical representation of the entire prompt becomes highly generalized. Standard dense embedding models calculate spatial proximity based on overall vector averages. As a result, the primetrade.ai and Banao Tech children matched closer to the dense query syntax, causing the Cross-Encoder filter to drop the IBM context slice out of the limited LLM window to maintain system boundary control.

---

##  4. Technical Roadmap: Scaling to Enterprise 9.9/10

To transform this solid production-ready prototype into an absolute enterprise-fault-tolerant architecture, the following engineering steps are on the roadmap:
1. **Hybrid Retrieval Implementation (Dense + Sparse Search):** Layering Qdrant semantic vector distances alongside an asynchronous sparse keyword tokenization framework (e.g., BM25). This guarantees that literal entities like "IBM" fire hard flags, bypassing semantic vector displacement.
2. **Pre-Query Payload Metadata Filtering:** Hardcoding specific runtime parameters inside the query statement context (Field == "Experience") to dynamically filter the Qdrant cluster space prior to executing mathematical cosine operations.
3. **Query Decomposition Workflow:** Introducing an asynchronous query parser that takes a complex compound prompt, shatters it into a multi-index array of distinct sub-queries, runs searches in parallel, and merges the resulting unified contexts.

---

##  5. Installation, Environments & Deployment Guide

Follow these steps sequentially to configure, link, and initialize your local production environment workspace.

### Step 5.1: Clone and Build Isolated Environment Architecture
Open an active shell instance in your workspace directory and execute the following initialization:
```bash
git clone https://github.com/Nazakath-Kanz/RAG-QA-System.git
cd RAG-QA-System

# Create isolated Python virtual runtime environment
python -m venv venv

# Activate Environment (Windows CMD/PowerShell)
venv\Scripts\activate

# Activate Environment (Linux / macOS Shells)
source venv/bin/activate
```

### Step 5.2: Install System Dependency Tree
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5.3: Set Environment Variables Profile
Create a configuration file named .env in the root directory context of your project:
```env
GEMINI_API_KEY=AIzaSyYourActualSecureGoogleGeminiApiKeyStringHere
EMBEDDING_MODEL=all-MiniLM-L6-v2
QDRANT_HOST=localhost
QDRANT_PORT=6333
LOGURU_LEVEL=INFO
```

---

##  6. Concurrent Microservice Execution Layout

Because this system splits responsibilities into distinct layers, you must run the server stack concurrently using two separate terminal windows with your virtual environment activated in both.

### Window 1: Fire up the Asynchronous FastAPI Backend Engine
```bash
uvicorn src.main:app --reload --port 8000
```
* **System Flag Verification:** Ensure your console outputs lines confirming: INFO: Uvicorn running on http://127.0.0.1:8000. You can instantly interact with your raw backend microservice layer or run manual query payloads via the Swagger OpenAPI UI at http://127.0.0.1:8000/docs.

### Window 2: Fire up the Dashboard Interface Frontend
```bash
streamlit run app.py --server.port 8501
```
* **System Flag Verification:** Your web browser will launch a reactive state-managed UI running on http://localhost:8501. Upload your target document profile, click Process & Index Data, and interact with your custom workspace.

---

##  7. Comprehensive Troubleshooting & Error Resolutions

If you hit any execution bugs during local instantiation, refer to this verified playbook:

###  Issue 1: TypeError: MarkdownMixin.markdown() got an unexpected keyword argument 'unsafe_allowed_html'
* **Why it happens:** A tiny past-tense syntax typo inside the Streamlit Markdown execution method (unsafe_allowed_html instead of the required unsafe_allow_html).
* **The Resolution:** Open app.py and ensure line 14 and line 107 match the singular keyword syntax format:
  ```python
  st.markdown("...", unsafe_allow_html=True)
  ```

###  Issue 2: Could not reach FastAPI backend server... Max retries exceeded with url /upload [WinError 10061]
* **Why it happens:** The Streamlit frontend UI is looking for the backend server engine on port 8000, but your FastAPI instance is offline, suspended, or its terminal tab was accidentally closed.
* **The Resolution:** Open a clean terminal shell window, navigate to your root project layout, activate venv, and explicitly restart the engine service:
  ```bash
  uvicorn src.main:app --reload
  ```

###  Issue 3: Stale Overlapping Vector Data Bleed (Old Data Cache)
* **Why it happens:** Making rapid file updates or repeatedly clicking index data causes new vectorized chunks to stack alongside stale database states inside the Qdrant local cluster memory layers.
* **The Resolution:** Open the file containing your upload route and clear collection blocks using the explicit asynchronous command before executing fresh data pushes:
  ```python
  await qdrant_client.delete_collection(collection_name="enterprise_parent_child_kb")
  ```

---

##  8. Advanced Enterprise Configurations (Optional Setup)

### Optional 8.1: Containerized Qdrant Infrastructure Setup
To execute your vector database isolated inside an independent container, bypass local setup by placing a docker-compose.yml file within your root project layer:
```yaml
version: '3.8'
services:
  qdrant_db:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_storage:/qdrant/storage
volumes:
  qdrant_storage:
```
To run the cluster, execute: docker-compose up -d

### Optional 8.2: Pre-Commit Automated Styling Hooks
Keep repository formatting spotless before data enters GitHub tracking stages by setting up automated linter systems:
```bash
pip install pre-commit black flake8 isort
pre-commit install
```