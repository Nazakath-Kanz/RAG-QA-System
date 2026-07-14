# create_stub.py
import os

stub_content = '''# Stub module - ChatVertexAI was removed from this langchain-community version.
# Ragas eagerly imports it even though we do not use Vertex AI (we use the
# direct Gemini API via langchain-google-genai instead). This stub exists
# only to satisfy that unused import path.

class ChatVertexAI:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "ChatVertexAI is not available - this project uses the Gemini API directly, not Vertex AI."
        )
'''

target_path = os.path.join("venv", "Lib", "site-packages", "langchain_community", "chat_models", "vertexai.py")

with open(target_path, "w", encoding="utf-8") as f:
    f.write(stub_content)

print(f"Stub created at: {target_path}")