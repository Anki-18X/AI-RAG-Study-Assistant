import os
import io
import requests
import streamlit as st

# -------------------------
# Config
# -------------------------

# Default to your Render backend, but allow override via env var
BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "https://ai-study-assistant-7oub.onrender.com",
).rstrip("/")


# -------------------------
# Helper functions
# -------------------------

def check_backend_health() -> bool:
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def build_index(files) -> dict:
    """
    Call /build_index on the FastAPI backend with uploaded PDFs.
    """
    if not files:
        return {"error": "Please upload at least one PDF."}

    # Prepare multipart form data for multiple files
    files_payload = []
    for f in files:
        # Reset pointer to start (important if file reused)
        f.seek(0)
        files_payload.append(
            (
                "files",
                (f.name, f.read(), "application/pdf"),
            )
        )

    try:
        resp = requests.post(
            f"{BACKEND_URL}/build_index",
            files=files_payload,
            timeout=300,
        )
    except Exception as e:
        return {"error": f"Failed to reach backend: {e}"}

    if resp.status_code != 200:
        try:
            data = resp.json()
        except Exception:
            data = {"detail": resp.text}
        return {
            "error": f"Backend error ({resp.status_code}): {data.get('detail', data)}"
        }

    return resp.json()


def ask_question(question: str) -> dict:
    """
    Call /ask on the FastAPI backend with the user's question.
    Backend auto-decides how many chunks to use (k is ignored).
    """
    if not question.strip():
        return {"error": "Please enter a question."}

    payload = {"question": question}

    try:
        resp = requests.post(
            f"{BACKEND_URL}/ask",
            json=payload,
            timeout=120,
        )
    except Exception as e:
        return {"error": f"Failed to reach backend: {e}"}

    if resp.status_code != 200:
        try:
            data = resp.json()
        except Exception:
            data = {"detail": resp.text}
        return {
            "error": f"Backend error ({resp.status_code}): {data.get('detail', data)}"
        }

    return resp.json()


# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(
    page_title="AI RAG Study Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 AI RAG Study Assistant")
st.write(
    "Upload your CSE notes as PDFs, build an index, and ask questions powered by Groq + Chroma + FastEmbed."
)

# Backend status
with st.sidebar:
    st.header("⚙️ Backend Settings")
    st.write(f"**Current backend:** `{BACKEND_URL}`")

    if st.button("Check backend health"):
        ok = check_backend_health()
        if ok:
            st.success("Backend is healthy ✅")
        else:
            st.error("Backend is not reachable ❌")

st.markdown("---")

# -------------------------
# Step 1: Upload PDFs & Build Index
# -------------------------

st.subheader("1️⃣ Upload PDFs & Build Index")

uploaded_files = st.file_uploader(
    "Upload one or more PDF files with your notes",
    type=["pdf"],
    accept_multiple_files=True,
    help="These will be processed, chunked automatically, and stored in the vector database.",
)

col_build_left, col_build_right = st.columns([1, 3])

with col_build_left:
    if st.button("🚀 Build / Rebuild Index"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF before building the index.")
        else:
            with st.spinner("Building index from your PDFs..."):
                result = build_index(uploaded_files)

            if "error" in result:
                st.error(result["error"])
            else:
                chunks = result.get("chunks", 0)
                st.success(f"Index built successfully ✅\n\n**Total chunks indexed:** {chunks}")

with col_build_right:
    st.info(
        "Every time you click **Build / Rebuild Index**, the previous index on the server "
        "is cleared and rebuilt from the PDFs you upload in this session."
    )

st.markdown("---")

# -------------------------
# Step 2: Ask Questions
# -------------------------

st.subheader("2️⃣ Ask Questions on Your Notes")

question = st.text_area(
    "Enter your question:",
    placeholder="e.g., Explain the differences between TCP and UDP, with examples.",
    height=100,
)

if st.button("❓ Ask"):
    with st.spinner("Thinking..."):
        result = ask_question(question)

    if "error" in result:
        st.error(result["error"])
    else:
        st.markdown("### ✅ Answer")
        st.write(result.get("answer", "No answer returned."))

        # Optional: show retrieved context & chunks for transparency
        with st.expander("🔍 Show retrieved context", expanded=False):
            st.code(result.get("context", ""), language="markdown")

        chunks = result.get("chunks", [])
        if chunks:
            with st.expander("📄 Show Retrieved Chunks", expanded=False):
                for i, ch in enumerate(chunks, start=1):
                    meta = ch.get("metadata", {}) or {}
                    src = meta.get("source_file", "unknown")
                    idx = meta.get("chunk_index", -1)
                    st.markdown(
                        f"**Chunk {i}**  \n"
                        f"*Source:* `{src}` — *chunk index:* `{idx}`"
                    )
                    st.write(ch.get("text", ""))
                    st.markdown("---")


st.markdown("---")
st.caption(
    "Built with ❤️ using FastAPI (Render backend), ChromaDB, FastEmbed, Groq LLM, and Streamlit."
)
