import os
import re
import io
import hashlib
from typing import List, Dict, Tuple

import fitz  # PyMuPDF
import faiss
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
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "Cyber Law GPT"
GOOGLE_DRIVE_FILE_ID = "1qv9f1Q3ILaa4ybYa85e8KyXbnlF8a2an"

# The supplied PDF is the knowledge source. If the file changes,
# update only the Drive file ID above.
PDF_URL = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"

# Small, lightweight sentence-transformers model suitable for
# Streamlit Cloud/Colab compared with large embedding models.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Groq model. The UI also lets the user select another model.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

MAX_PDF_BYTES = 25 * 1024 * 1024
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180
TOP_K_DEFAULT = 5


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }
    .law-badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        background: rgba(100, 100, 100, 0.12);
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
    }
    .source-box {
        border-left: 4px solid #888;
        padding: 0.7rem 1rem;
        margin: 0.5rem 0;
        background: rgba(127, 127, 127, 0.07);
        border-radius: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Helpers
# ============================================================

def get_groq_api_key() -> str:
    """Read Groq API key from Streamlit secrets or environment."""
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        key = ""

    return key or os.getenv("GROQ_API_KEY", "")


def download_pdf(url: str) -> bytes:
    """Download the configured law PDF safely into memory."""
    response = requests.get(
        url,
        timeout=60,
        allow_redirects=True,
        headers={"User-Agent": "Cyber-Law-GPT/1.0"},
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    data = response.content

    # Google Drive can return an HTML confirmation page for some files.
    # A real PDF should begin with the %PDF signature.
    if not data.startswith(b"%PDF"):
        raise ValueError(
            "The Google Drive link did not return a PDF file. "
            "Make sure the Drive file is shared as 'Anyone with the link'."
        )

    if len(data) > MAX_PDF_BYTES:
        raise ValueError("The configured PDF is larger than the 25 MB safety limit.")

    return data


def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Create overlapping character-based chunks."""
    text = clean_text(text)
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]

        # Prefer ending at a sentence/line boundary when possible.
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
    """Extract text page-by-page so sources can report page numbers."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []

    try:
        for page_number, page in enumerate(doc, start=1):
            text = clean_text(page.get_text("text"))
            if text:
                pages.append(
                    {
                        "page": page_number,
                        "text": text,
                    }
                )
    finally:
        doc.close()

    if not pages:
        raise ValueError(
            "No selectable text was found in the PDF. "
            "This version expects a text-based PDF."
        )

    return pages


def build_chunks(pages: List[Dict]) -> List[Dict]:
    records = []

    for page in pages:
        chunks = chunk_text(page["text"])
        for index, chunk in enumerate(chunks):
            records.append(
                {
                    "page": page["page"],
                    "chunk": index + 1,
                    "text": chunk,
                }
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

    return index


@st.cache_resource(show_spinner=False)
def prepare_knowledge_base(pdf_bytes: bytes):
    pages = extract_pdf_pages(pdf_bytes)
    records = build_chunks(pages)
    model = load_embedding_model()
    index = build_faiss_index(records, model)

    return records, model, index


def retrieve(
    question: str,
    records: List[Dict],
    model,
    index,
    top_k: int,
    min_score: float,
) -> List[Dict]:
    """Semantic retrieval using cosine similarity via normalized FAISS IP."""
    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
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
            f"[SOURCE {i} | Page {item['page']} | Chunk {item['chunk']}]\n"
            f"{item['text']}"
        )

    return "\n\n".join(parts)


def build_system_prompt(technicality: str, response_size: str) -> str:
    technicality_instructions = {
        "Simple": (
            "Use plain language suitable for a general reader. "
            "Briefly explain legal terms when they appear."
        ),
        "Balanced": (
            "Use clear legal terminology but explain important terms in plain language."
        ),
        "Technical / Legal": (
            "Use precise legal terminology, identify relevant provisions where "
            "supported by the retrieved text, and distinguish legal rules from interpretation."
        ),
    }

    size_instructions = {
        "Short": "Keep the answer concise, usually 3–6 short paragraphs or bullets.",
        "Medium": "Give a practical, structured answer with the important details.",
        "Detailed": "Give a comprehensive answer, including applicable provisions, reasoning, exceptions, and source pages when supported.",
    }

    return f"""
You are Cyber Law GPT, a Pakistan-focused legal-information RAG assistant.

KNOWLEDGE BOUNDARY:
- The retrieved context below is the primary and controlling knowledge source for this answer.
- Answer questions about Pakistani cyber/electronic-crime law only when the retrieved context supports the answer.
- Do NOT invent sections, penalties, definitions, authorities, procedures, case law, dates, or legal conclusions.
- If the answer is not supported by the retrieved context, clearly say:
  "Sorry, this information is not found in the provided cyber-law document."
- Do not use your general memory to fill missing legal information.
- When the question is unrelated to the provided cyber-law document, say that it is outside the document's scope.

LEGAL-SAFETY RULES:
- This is legal information, not a substitute for advice from a qualified Pakistani lawyer.
- Do not encourage, facilitate, or provide operational instructions for illegal cyber activity,
  unauthorized access, credential theft, malware deployment, evasion, fraud, harassment,
  privacy violations, or other wrongdoing.
- If a user asks how to commit or facilitate a cyber offence, explain the relevant legal risk
  from the retrieved document and, where appropriate, suggest lawful defensive/security practices.
- Do not make a person-specific determination of guilt or innocence.
- Do not claim that an offence definitely occurred based only on a short hypothetical.
- Clearly distinguish what the document says from practical/legal interpretation.
- Do not request passwords, OTPs, CNIC numbers, banking credentials, private keys, or other secrets.

ANSWER STYLE:
{technicality_instructions[technicality]}
{size_instructions[response_size]}

CITATION REQUIREMENT:
- Use source markers like [Page 12] when making a claim supported by the retrieved context.
- Only cite pages actually present in the retrieved context.
- Do not fabricate citations.
"""


def ask_groq(
    question: str,
    context: str,
    technicality: str,
    response_size: str,
    model_name: str,
    temperature: float,
) -> str:
    api_key = get_groq_api_key()

    if not api_key:
        raise RuntimeError(
            "Groq API key not found. Add GROQ_API_KEY to Streamlit Secrets "
            "or set it as an environment variable."
        )

    client = Groq(api_key=api_key)

    system_prompt = build_system_prompt(technicality, response_size)

    user_prompt = f"""
RETRIEVED LEGAL CONTEXT:
{context}

USER QUESTION:
{question}

Answer only from the retrieved legal context. If the context does not support the answer,
use the required "not found" response instead of guessing.
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )

    return response.choices[0].message.content.strip()


def source_signature(item: Dict) -> str:
    return hashlib.sha256(
        f"{item['page']}|{item['chunk']}|{item['text']}".encode("utf-8")
    ).hexdigest()[:10]


# ============================================================
# Startup knowledge base
# ============================================================

st.title("⚖️ Cyber Law GPT")
st.markdown(
    '<span class="law-badge">Pakistan Cyber-Law RAG Assistant</span>',
    unsafe_allow_html=True,
)

st.caption(
    "Ask questions about the Pakistani cyber-law document. "
    "Answers are grounded in the supplied PDF and show the relevant source pages."
)

with st.sidebar:
    st.header("⚙️ Response Settings")

    technicality = st.selectbox(
        "Technicality level",
        ["Simple", "Balanced", "Technical / Legal"],
        index=1,
        help="Controls vocabulary and depth of legal terminology.",
    )

    response_size = st.selectbox(
        "Response size",
        ["Short", "Medium", "Detailed"],
        index=1,
    )

    top_k = st.slider(
        "Retrieved sources",
        min_value=2,
        max_value=8,
        value=TOP_K_DEFAULT,
        help="Number of semantically relevant chunks sent to the LLM.",
    )

    min_score = st.slider(
        "Retrieval confidence",
        min_value=0.10,
        max_value=0.75,
        value=0.30,
        step=0.05,
        help="Higher values make the assistant stricter about whether the PDF supports an answer.",
    )

    temperature = st.slider(
        "Creativity",
        min_value=0.0,
        max_value=0.6,
        value=0.1,
        step=0.1,
        help="Lower values are recommended for legal-information tasks.",
    )

    model_name = st.selectbox(
        "Groq model",
        [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        ],
        index=0,
    )

    st.divider()

    st.markdown("**Knowledge source**")
    st.caption("The configured Google Drive PDF is downloaded and embedded automatically on startup.")

    st.markdown("**Privacy & security**")
    st.caption(
        "Do not enter passwords, OTPs, CNIC numbers, bank details, private keys, "
        "or other confidential information."
    )


# Download + index creation happen automatically on startup and are cached.
try:
    with st.spinner("Loading Pakistani cyber-law document and building embeddings..."):
        pdf_bytes = download_pdf(PDF_URL)
        records, embedding_model, faiss_index = prepare_knowledge_base(pdf_bytes)

    st.success(
        f"Knowledge base ready — {len(records)} searchable chunks indexed."
    )
except Exception as exc:
    st.error(f"Knowledge-base startup error: {exc}")
    st.info(
        "For Google Drive, ensure the PDF is shared as 'Anyone with the link'. "
        "You can also replace the Drive file ID in app.py."
    )
    st.stop()


# ============================================================
# Chat
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input(
    "Ask a question about Pakistani cyber law..."
)

if question:
    question = question.strip()

    if not question:
        st.stop()

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the law document..."):
            results = retrieve(
                question=question,
                records=records,
                model=embedding_model,
                index=faiss_index,
                top_k=top_k,
                min_score=min_score,
            )

        if not results:
            answer = (
                "Sorry, this information is not found in the provided cyber-law document."
            )
            st.markdown(answer)
            st.session_state.last_sources = []
        else:
            context = format_context(results)

            try:
                with st.spinner("Generating a grounded legal-information response..."):
                    answer = ask_groq(
                        question=question,
                        context=context,
                        technicality=technicality,
                        response_size=response_size,
                        model_name=model_name,
                        temperature=temperature,
                    )
            except Exception as exc:
                st.error(f"Groq error: {exc}")
                st.stop()

            st.markdown(answer)

            st.session_state.last_sources = results

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )


# ============================================================
# Sources for the latest answer
# ============================================================

if st.session_state.last_sources:
    st.divider()
    st.subheader("📚 RAG Sources")

    for item in st.session_state.last_sources:
        st.markdown(
            f"""
            <div class="source-box">
                <strong>Page {item['page']} · Chunk {item['chunk']}</strong>
                &nbsp;·&nbsp; Similarity: {item['score']:.3f}
                <br><br>
                {item['text'][:700]}{"..." if len(item['text']) > 700 else ""}
                <br><br>
                <code>Source ID: {source_signature(item)}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()
st.caption(
    "⚠️ Legal-information disclaimer: Cyber Law GPT is an educational RAG tool, "
    "not a law firm, lawyer, or substitute for professional legal advice. "
    "The authoritative legal text should be checked before relying on an answer."
)
