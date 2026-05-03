import streamlit as st
import io
import numpy as np

# -------- FAISS SAFE IMPORT --------
try:
    import faiss
except:
    import faiss_cpu as faiss

import google.generativeai as genai

st.set_page_config(page_title="RAG Demo (Gemini + FAISS)", layout="centered")
st.title("📚 RAG Demo (Gemini + FAISS)")

# ---------------- CONFIG ----------------
st.sidebar.header("🔑 API Key")
gemini_api = st.sidebar.text_input("Gemini API Key", type="password")

EMBED_MODEL = "models/text-embedding-004"
GEN_MODEL = "gemini-2.5-flash"

if gemini_api:
    genai.configure(api_key=gemini_api)

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

# ---------------- EMBEDDING ----------------
def embed_texts(texts):
    embeddings = []

    for t in texts:
        res = genai.embed_content(
            model=EMBED_MODEL,
            content=t
        )
        embeddings.append(res["embedding"])

    emb_array = np.array(embeddings).astype("float32")

    if len(emb_array.shape) != 2:
        st.error("Embedding failed")
        st.stop()

    return emb_array

# ---------------- FAISS ----------------
def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# ---------------- RETRIEVAL ----------------
def retrieve(query, index, chunks, k=4):
    q_emb = embed_texts([query])[0].reshape(1, -1)
    _, I = index.search(q_emb, k)

    results = []
    for idx in I[0]:
        if 0 <= idx < len(chunks):
            results.append(chunks[idx])

    return results

# ---------------- GENERATION ----------------
def generate_answer(query, contexts):
    model = genai.GenerativeModel(GEN_MODEL)

    context_block = "\n\n---\n\n".join(contexts)

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
    if not gemini_api:
        st.error("Enter Gemini API key")
    elif not files:
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

        st.info("Generating embeddings...")
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
    if not gemini_api:
        st.error("Enter Gemini API key")
    elif st.session_state.index is None:
        st.warning("Build index first")
    elif not query:
        st.warning("Enter a query")
    else:
        st.info("Retrieving context...")
        contexts = retrieve(query, st.session_state.index, st.session_state.chunks, k)

        st.markdown("### 🔎 Retrieved Context")
        for i, c in enumerate(contexts, 1):
            st.markdown(f"**Chunk {i}:** {c[:300]}...")

        st.info("Generating answer...")
        answer = generate_answer(query, contexts)

        st.markdown("### 🤖 Answer")
        st.write(answer)
