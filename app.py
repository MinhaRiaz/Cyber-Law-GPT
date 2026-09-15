import hashlib
import io
import os
import re
from typing import Dict, List, Tuple

import faiss
import fitz  # PyMuPDF
import numpy as np
import requests
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

# ============================================================
# Cyber Law GPT
# Pakistan-focused RAG assistant
# ============================================================

st.set_page_config(
    page_title="Cyber Law GPT",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "CyberLawGPT"
GOOGLE_DRIVE_FILE_ID = "1qv9f1Q3ILaa4ybYa85e8KyXbnlF8a2an"
PDF_URL = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

MAX_PDF_BYTES = 25 * 1024 * 1024
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180

# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    .metric-card {
        text-align: center;
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Helpers & Processing
# ============================================================


def get_groq_api_key() -> str:
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        key = ""
    return key or os.getenv("GROQ_API_KEY", "")


def download_pdf(url: str) -> bytes:
    response = requests.get(
        url,
        timeout=60,
        allow_redirects=True,
        headers={"User-Agent": "Cyber-Law-GPT/1.0"},
    )
    response.raise_for_status()
    data = response.content

    if not data.startswith(b"%PDF"):
        raise ValueError("The Drive link did not return a valid PDF file.")

    if len(data) > MAX_PDF_BYTES:
        raise ValueError("PDF exceeds maximum limit of 25 MB.")

    return data


def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> List[str]:
    text = clean_text(text)
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]

        if end < len(text):
            candidates = [
                chunk.rfind("\n"),
                chunk.rfind(". "),
                chunk.rfind("; "),
                chunk.rfind(" "),
            ]
            boundary = max(candidates)
            if boundary >= int(chunk_size * 0.55):
                end = start + boundary + 1
                chunk = text[start:end]

        chunks.append(chunk.strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return [c for c in chunks if c]


@st.cache_data(show_spinner=False)
def extract_pdf_pages(pdf_bytes: bytes) -> List[Dict]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    try:
        for page_number, page in enumerate(doc, start=1):
            text = clean_text(page.get_text("text"))
            if text:
                pages.append({"page": page_number, "text": text})
    finally:
        doc.close()

    if not pages:
        raise ValueError("No text extracted from PDF.")
    return pages


def build_chunks(pages: List[Dict]) -> List[Dict]:
    records = []
    for page in pages:
        chunks = chunk_text(page["text"])
        for index, chunk in enumerate(chunks):
            records.append(
                {"page": page["page"], "chunk": index + 1, "text": chunk}
            )
    return records


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def build_faiss_index(records: List[Dict], model):
    texts = [r["text"] for r in records]
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index, dimension


@st.cache_resource(show_spinner=False)
def prepare_knowledge_base(pdf_bytes: bytes):
    pages = extract_pdf_pages(pdf_bytes)
    records = build_chunks(pages)
    model = load_embedding_model()
    index, dimension = build_faiss_index(records, model)
    return records, model, index, len(pages), dimension


def retrieve(
    question: str, records: List[Dict], model, index, top_k: int, min_score: float
) -> List[Dict]:
    query_embedding = model.encode(
        [question], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")

    search_k = min(max(top_k * 3, top_k), len(records))
    scores, indices = index.search(query_embedding, search_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        score = float(score)
        if score >= min_score:
            result = dict(records[idx])
            result["score"] = score
            results.append(result)
        if len(results) >= top_k:
            break
    return results


def format_context(results: List[Dict]) -> str:
    parts = []
    for i, item in enumerate(results, start=1):
        parts.append(
            f"[SOURCE {i} | Page {item['page']} | Chunk {item['chunk']}]\n{item['text']}"
        )
    return "\n\n".join(parts)


def ask_groq(
    question: str,
    context: str,
    technicality: str,
    response_size: str,
    language: str,
    legal_focus: str,
    model_name: str,
) -> str:
    api_key = get_groq_api_key()
    if not api_key:
        raise RuntimeError("Groq API key missing.")

    client = Groq(api_key=api_key)

    system_prompt = f"""
You are CyberLawGPT, a assistant for Pakistani cyber law.
- Technicality level: {technicality}
- Response size requirement: {response_size}
- Language: {language}
- Focus: {legal_focus}

Base answers strictly on the retrieved context below. If unsupported, inform the user that the answer was not found in the legal context provided.
"""

    user_prompt = f"RETRIEVED CONTEXT:\n{context}\n\nQUESTION:\n{question}"

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


# ============================================================
# Sidebar Settings
# ============================================================

with st.sidebar:
    st.title("⚙️ CyberLawGPT Settings")
    st.caption("Version 2.0.0")

    technicality = st.selectbox(
        "Technicality level",
        ["Beginner", "Intermediate", "Technical / Legal"],
        index=1,
    )

    response_size = st.select_slider(
        "Response size",
        options=["Short", "Medium", "Very detailed"],
        value="Medium",
    )

    language = st.selectbox("Answer language", ["English", "Urdu"], index=0)

    legal_focus = st.selectbox(
        "Legal focus", ["General Pakistani cyber law"], index=0
    )

    st.divider()
    st.markdown("🔍 **Retrieval**")

    top_k = st.slider("Number of passages", min_value=1, max_value=10, value=5)

    min_score = st.slider(
        "Minimum similarity",
        min_value=0.05,
        max_value=0.80,
        value=0.25,
        step=0.05,
    )

    st.divider()
    st.markdown("🤖 **Groq**")

    model_name = st.selectbox(
        "Model",
        ["llama-3.3-70b-versatile", "llama3-8b-8192", "mixtral-8x7b-32768"],
        index=0,
    )

    show_sources_option = st.checkbox("Show retrieved sources", value=True)

    st.divider()
    st.markdown("🔐 **Privacy**")
    st.caption(
        "Do not enter passwords, API keys, CNIC numbers, financial information, "
        "private credentials, or unnecessary confidential case information.\n\n"
        "Questions are sent to Groq for response generation."
    )

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# Knowledge Base Initialization
# ============================================================

try:
    pdf_bytes = download_pdf(PDF_URL)
    records, embedding_model, faiss_index, num_pages, vector_dim = (
        prepare_knowledge_base(pdf_bytes)
    )
except Exception as exc:
    st.error(f"Error initializing knowledge base: {exc}")
    st.stop()


# ============================================================
# Main UI Metrics Header
# ============================================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("<span style='color: #6c757d; font-size: 0.85rem;'>PDF pages</span>", unsafe_allow_html=True)
    st.markdown(f"### {num_pages}")
with col2:
    st.markdown("<span style='color: #6c757d; font-size: 0.85rem;'>Indexed chunks</span>", unsafe_allow_html=True)
    st.markdown(f"### {len(records)}")
with col3:
    st.markdown("<span style='color: #6c757d; font-size: 0.85rem;'>Vector dimension</span>", unsafe_allow_html=True)
    st.markdown(f"### {vector_dim}")
with col4:
    st.markdown("<span style='color: #6c757d; font-size: 0.85rem;'>Retrieval</span>", unsafe_allow_html=True)
    st.markdown("### FAISS")

st.divider()

# Prompt suggestion list
prompt_suggestions = [
    "What is unauthorized access under Pakistani cyber law?",
    "Explain cybercrime law in simple language.",
    "What are the legal consequences of unauthorized access?",
    "What should a victim do after an online cybercrime incident?",
    "Explain a relevant provision of Pakistani cyber law.",
    "What is the difference between cyber fraud and unauthorized access?",
]

if "selected_prompt" not in st.session_state:
    st.session_state.selected_prompt = None

# Render suggestions if conversation hasn't started
if "messages" not in st.session_state or len(st.session_state.messages) == 0:
    st.markdown("💡 **Try asking**")
    grid_cols = st.columns(2)
    for idx, prompt in enumerate(prompt_suggestions):
        col_target = grid_cols[idx % 2]
        if col_target.button(prompt, key=f"btn_{idx}", use_container_width=True):
            st.session_state.selected_prompt = prompt

# Session message state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"] and show_sources_option:
            with st.expander("📚 Retrieved legal sources"):
                for source in msg["sources"]:
                    st.write(
                        f"**Page {source['page']} (Chunk {source['chunk']})** - Similarity: {source['score']:.2f}"
                    )
                    st.caption(source["text"])

# Get user input either from chat bar or prompt suggestion
user_input = st.chat_input("Ask CyberLawGPT about Pakistani cyber law...")
if st.session_state.selected_prompt:
    user_input = st.session_state.selected_prompt
    st.session_state.selected_prompt = None

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        results = retrieve(
            question=user_input,
            records=records,
            model=embedding_model,
            index=faiss_index,
            top_k=top_k,
            min_score=min_score,
        )

        if not results:
            answer = "I could not find relevant information in the provided Pakistani cyber-law document."
            st.markdown(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": []}
            )
        else:
            context = format_context(results)
            try:
                answer = ask_groq(
                    question=user_input,
                    context=context,
                    technicality=technicality,
                    response_size=response_size,
                    language=language,
                    legal_focus=legal_focus,
                    model_name=model_name,
                )
            except Exception as exc:
                answer = f"I could not generate the answer because the Groq request failed.\n\nPlease verify your GROQ_API_KEY, model selection, internet connection, and Groq API availability. ({exc})"

            st.markdown(answer)

            if show_sources_option:
                with st.expander("📚 Retrieved legal sources"):
                    for source in results:
                        st.write(
                            f"**Page {source['page']} (Chunk {source['chunk']})** - Similarity: {source['score']:.2f}"
                        )
                        st.caption(source["text"])

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": results}
            )

st.divider()
st.caption(
    "CyberLawGPT • Pakistani cyber-law RAG • FAISS • Sentence Transformers • Groq • Legal information only, not legal advice."
)
