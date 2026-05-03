import streamlit as st
import io
import numpy as np

# -------- FAISS SAFE IMPORT --------
try:
    import faiss
except:
    import faiss_cpu as faiss

# -------- LOCAL EMBEDDINGS --------
from sentence_transformers import SentenceTransformer

# -------- GEMINI --------
import google.generativeai as genai

st.set_page_config(page_title="RAG Demo (Cosine Similarity)", layout="centered")
st.title("📚 RAG Demo (Cosine Similarity + Scores)")

# ---------------- CONFIG ----------------
st.sidebar.header("🔑 API Key")
gemini_api = st.sidebar.text_input("Gemini API Key", type="password")

GEN_MODEL = "gemini-2.5-flash"

if gemini_api:
    genai.configure(api_key=gemini_api)

# -------- LOAD EMBEDDING MODEL --------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ---------------- FILE READING ----------------
def read_files(files):
    docs = []
    for f in files:
        name = f.name
        content = ""

        try:
            if name.lower().endswith(".pdf"):
                from PyPDF2 import PdfReader
                reader = PdfReader(io.BytesIO(f.read()))
                content = "\n".join([p.extract_text() or "" for p in reader.pages])
            else:
                content = f.read().decode("utf-8", errors="ignore")
        except:
            content = ""

        if content.strip():
            docs.append((name, content))

    return docs

# ---------------- CHUNKING ----------------
def chunk_text(text, size=800, overlap=120):
    chunks = []
    start = 0

    while start < len(text):
        chunk = text[start:start + size]
        if chunk.strip():
            chunks.append(chunk)
        start += size - overlap

    return chunks

# ---------------- EMBEDDING (COSINE READY) ----------------
def embed_texts(texts):
    embeddings = embedder.encode(texts)

    # normalize → cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    return np.array(embeddings).astype("float32")

# ---------------- FAISS ----------------
def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity
    index.add(embeddings)
    return index

# ---------------- RETRIEVAL ----------------
def retrieve(query, index, chunks, k=4):
    q_emb = embed_texts([query])
    D, I = index.search(q_emb, k)

    results = []
    for score, idx in zip(D[0], I[0]):
        if 0 <= idx < len(chunks):
            results.append({
                "text": chunks[idx],
                "score": float(score)
            })

    return results, q_emb

# ---------------- GENERATION ----------------
def generate_answer(query, contexts):
    model = genai.GenerativeModel(GEN_MODEL)

    context_block = "\n\n---\n\n".join([c["text"] for c in contexts])

    prompt = f"""
You are a helpful assistant.

Answer ONLY using the provided context.
If the answer is not present, say "I don't know".

Context:
{context_block}

Question:
{query}

Be concise and natural.
"""

    return model.generate_content(prompt).text

# ---------------- SESSION ----------------
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = []

# ---------------- UI ----------------
st.header("1️⃣ Upload Documents")

files = st.file_uploader(
    "Upload PDF / TXT / MD",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True
)

if st.button("Build Index"):
    if not files:
        st.warning("Upload files first")
    else:
        st.info("Reading documents...")
        docs = read_files(files)

        if not docs:
            st.error("No readable content")
            st.stop()

        all_chunks = []

        for name, text in docs:
            chunks = chunk_text(text)
            all_chunks.extend([f"[{name}] {c}" for c in chunks])

        if not all_chunks:
            st.error("No chunks created")
            st.stop()

        st.info(f"Created {len(all_chunks)} chunks")

        st.info("Generating embeddings (cosine normalized)...")
        embeddings = embed_texts(all_chunks)

        st.info("Building FAISS index...")
        index = build_index(embeddings)

        st.session_state.index = index
        st.session_state.chunks = all_chunks

        st.success("Index ready!")

# ---------------- QUERY ----------------
st.header("2️⃣ Ask Questions")

query = st.text_input("Enter your question")
k = st.slider("Top-K results", 2, 8, 4)

if st.button("Search & Answer"):
    if st.session_state.index is None:
        st.warning("Build index first")
    elif not query:
        st.warning("Enter a query")
    elif not gemini_api:
        st.error("Enter Gemini API key")
    else:
        st.info("Retrieving context...")
        contexts, q_emb = retrieve(query, st.session_state.index, st.session_state.chunks, k)

        st.markdown("### 🔎 Retrieved Context (Cosine Scores)")
        for i, r in enumerate(contexts, 1):
            st.markdown(f"""
**Chunk {i} (Score: {r['score']:.4f})**  
{r['text'][:300]}...
""")

        # vector preview (first 10 values)
        st.markdown("### 🧠 Query Vector (first 10 values)")
        st.write(q_emb[0][:10])

        st.info("Generating answer...")
        answer = generate_answer(query, contexts)

        st.markdown("### 🤖 Answer")
        st.write(answer)
