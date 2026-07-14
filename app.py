# app.py
import streamlit as st
import requests

# Set page configuration with a modern AI dashboard layout
st.set_page_config(
    page_title="Enterprise RAG Dashboard", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS styling for a dark-mode minimalist AI aesthetic
st.markdown("""
    <style>
    .main { background-color: #0E1117; color: #FAFAFA; }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; border-radius: 8px;
        padding: 10px 24px; font-weight: bold; transition: all 0.3s ease;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(102,126,234,0.4); }
    .context-card {
        background-color: #1F2937; border-left: 4px solid #6366F1;
        padding: 15px; border-radius: 6px; margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR: INGESTION LAYER -----------------
with st.sidebar:
    st.markdown("<h2 style='color: #6366F1;'>📁 Knowledge Layer</h2>", unsafe_allow_html=True)
    st.markdown("Stream raw documents natively into your asynchronous Qdrant vector space matrix.")
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "Upload Technical Resume / Documents", 
        type=["pdf", "txt"],
        help="Supports raw text streams and multi-page binary PDF extractions."
    )
    
    if uploaded_file is not None:
        st.info(f"⚡ Ready to ingest: `{uploaded_file.name}`")
        if st.button("🚀 Process & Index Data", use_container_width=True):
            with st.spinner("Executing chunking splitters & computing vectors..."):
                # Read the file bytes directly into a stream payload
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    # Point straight to your running FastAPI upload route
                    response = requests.post("http://127.0.0.1:8000/upload", files=files)
                    if response.status_code == 200:
                        st.success(f"Successfully committed '{uploaded_file.name}' to Qdrant cluster!")
                        st.balloons()
                    else:
                        st.error(f"Ingestion failed: {response.json().get('detail')}")
                except Exception as e:
                    st.error(f"Could not reach FastAPI backend server: {e}")

# ----------------- MAIN PANEL: CHAT ARENA -----------------
st.markdown("<h1 style='text-align: center; font-weight: 800;'>🚀 Enterprise Parent-Child RAG</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #9CA3AF;'>Hybrid Retrieval Space &bull; FlashRank Cross-Encoder Reranking &bull; Gemini Inference Synthesis</p>", unsafe_allow_html=True)
st.markdown("---")

# Maintain state history across app redraw runs
if "messages" not in st.session_state:
    st.session_state.messages = []
if "latest_contexts" not in st.session_state:
    st.session_state.latest_contexts = []

# Split layout: 70% Chat Interface, 30% Real-time Database Diagnostics Context Viewer
col_chat, col_diagnostics = st.columns([0.65, 0.35], gap="large")

with col_chat:
    st.subheader("💬 Interactive Inference Arena")
    
    # Render historical chat logs smoothly
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Mode toggle — lets the user choose between the base pipeline and the
    # self-correcting agentic pipeline (query rewrite + retry on low confidence)
    agentic_mode = st.toggle("🤖 Agentic Mode (auto-retry on failed retrieval)", value=False)

    # Interactive input area
    if question := st.chat_input("Query anything regarding Kanz's architectural metrics..."):
        # Display User query instantly
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        endpoint = "http://127.0.0.1:8000/ask_agentic" if agentic_mode else "http://127.0.0.1:8000/ask"
        spinner_text = (
            "Routing query -> Retrieve -> Generate -> Confidence Check -> Retry if needed..."
            if agentic_mode
            else "Routing query -> Database Vectors -> FlashRank Rerank -> LLM Generation..."
        )

        # Call Backend for Inference Synthesis
        with st.chat_message("assistant"):
            with st.spinner(spinner_text):
                try:
                    res = requests.post(endpoint, json={"question": question})
                    if res.status_code == 200:
                        data = res.json()
                        answer = data.get("answer")
                        contexts = data.get("context_used", [])
                        
                        # Render synthesized text answer
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                        # Save contexts into session state to display in side pane
                        st.session_state.latest_contexts = contexts
                        st.rerun()  # Forces layout refresh to display context updates immediately
                    else:
                        st.error(" Downstream pipeline failure yielding response tokens.")
                except Exception as e:
                    st.error(f"Inference cluster routing fault: {e}")

# ----------------- RIGHT COLUMN: REAL-TIME DIAGNOSTICS -----------------
with col_diagnostics:
    st.markdown("<h3 style='color: #A78BFA;'>🔍 Vector Context Diagnostics</h3>", unsafe_allow_html=True)
    st.markdown("Displays the high-density Parent text blocks pulled by your vector search and ranked by FlashRank.")
    
    if st.session_state.latest_contexts:
        for idx, ctx in enumerate(st.session_state.latest_contexts):
            st.markdown(f"""
                <div class="context-card">
                    <strong style="color: #6366F1;">📍 Retrieved Context Block {idx + 1}</strong>
                    <p style="font-size: 0.9rem; margin-top: 5px; color: #E5E7EB;">{ctx}</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No query executed yet. Ask a question to view real-time vector matches.")