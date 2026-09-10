# ⚖️ Cyber Law GPT

**Cyber Law GPT** is a Pakistan-focused Retrieval-Augmented Generation (RAG) application built with:

- Python
- Streamlit
- FAISS
- Sentence Transformers
- Groq API
- PyMuPDF

It downloads the configured Pakistani cyber-law PDF from Google Drive on startup, extracts the text page-by-page, creates semantic embeddings, stores them in a FAISS vector index, retrieves the most relevant legal passages for each question, and sends only those retrieved passages to Groq for answer generation.

## Important legal scope

The application is intentionally **document-grounded**.

It should answer a question only when the supplied cyber-law document contains enough information to support the answer. If the relevant information is not retrieved, the application returns:

> Sorry, this information is not found in the provided cyber-law document.

This avoids making up legal provisions, penalties, sections, procedures, or interpretations.

The application also contains a safety-oriented system prompt that does not provide operational assistance for unauthorized access, malware, credential theft, fraud, evasion, or other cyber wrongdoing. For such questions, it should provide lawful/legal-risk information when supported by the document.

## Knowledge source

The application is configured to download the PDF from the Google Drive file supplied for this project.

The Drive file must be accessible with:

**Anyone with the link → Viewer**

If the file is private, Streamlit Cloud/Colab will not be able to download it.

The current file ID is stored in `app.py`:

```python
GOOGLE_DRIVE_FILE_ID = "1qv9f1Q3ILaa4ybYa85e8KyXbnlF8a2an"
```

## Project structure

Only these three project files are required:

```text
Cyber-Law-GPT/
│
├── app.py
├── requirements.txt
└── readme.md
```

The PDF does **not** need to be committed to GitHub because the app downloads it automatically on startup.

## Features

### RAG pipeline

```text
Google Drive PDF
       ↓
Download on startup
       ↓
PyMuPDF text extraction
       ↓
Page-aware chunking
       ↓
Sentence Transformer embeddings
       ↓
FAISS vector index
       ↓
User question
       ↓
Semantic retrieval
       ↓
Relevant legal chunks
       ↓
Groq LLM
       ↓
Grounded answer + source pages
```

### UI controls

The sidebar includes:

- **Technicality level**
  - Simple
  - Balanced
  - Technical / Legal

- **Response size**
  - Short
  - Medium
  - Detailed

- **Retrieved sources**
  - Controls the number of RAG chunks supplied to the LLM.

- **Retrieval confidence**
  - Controls the minimum semantic-similarity threshold.

- **Creativity**
  - Temperature control.
  - A low value is recommended for legal-information questions.

- **Groq model**
  - `openai/gpt-oss-120b`
  - `openai/gpt-oss-20b`

The code uses the Groq model selected by the user.

## 1. Get a Groq API key

Create a Groq API key from your Groq account.

Do **not** put the API key directly into `app.py`.

### Streamlit Cloud

In your Streamlit Cloud application settings:

```text
Settings
→ Secrets
```

Add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

### Colab

Set the environment variable before starting Streamlit:

```python
import os
os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"
```

For better security, use Colab Secrets rather than hard-coding the key in a notebook.

## 2. Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your Groq key:

### Windows PowerShell

```powershell
$env:GROQ_API_KEY="your_groq_api_key_here"
```

Run:

```bash
streamlit run app.py
```

## 3. Run in Google Colab

Upload or clone the three files into Colab.

Install dependencies:

```python
!pip install -r requirements.txt
```

Set the Groq key:

```python
import os
os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"
```

Then run:

```python
!streamlit run app.py &>/content/streamlit.log &
```

You can expose Streamlit through your preferred Colab-compatible tunneling method.

## 4. Deploy on Streamlit Cloud

Push:

```text
app.py
requirements.txt
readme.md
```

to a GitHub repository.

Then create a Streamlit app using:

```text
app.py
```

Add the secret:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

The application will download the law PDF and build the FAISS index on startup.

## Why FAISS?

FAISS provides fast vector similarity search.

The application uses normalized embeddings and:

```python
faiss.IndexFlatIP
```

Inner product on normalized vectors behaves as cosine similarity, which is useful for semantic retrieval.

## Why page-aware chunks?

Every chunk keeps:

- PDF page number
- chunk number
- extracted text

This allows the UI to show the user where the retrieved legal information came from.

Example:

```text
Page 12 · Chunk 2
Similarity: 0.713
```

## Security design

The project follows basic secure-application practices:

1. **No API key in source code**
   - Uses Streamlit Secrets/environment variables.

2. **No credentials requested from users**
   - The UI warns users not to enter passwords, OTPs, CNIC numbers, bank information, or private keys.

3. **RAG grounding**
   - The LLM receives retrieved legal context instead of being asked to freely invent legal answers.

4. **Out-of-scope handling**
   - Unsupported questions should return a not-found response rather than hallucinating.

5. **Cyber-abuse guardrail**
   - The system prompt does not provide operational instructions for cyber abuse.

6. **PDF size limit**
   - Startup downloads are limited to 25 MB.

7. **Low-temperature legal responses**
   - The default temperature is deliberately low.

8. **Source visibility**
   - Retrieved pages and similarity scores are shown under the answer.

## Legal disclaimer

Cyber Law GPT is an **educational/legal-information RAG application**.

It is not a lawyer, law firm, court, government authority, or substitute for professional legal advice.

Cyber/electronic-crime laws can be amended, interpreted by courts, supplemented by rules/regulations, or affected by other legislation. Before relying on an answer for an actual legal matter, verify the current official legal text and consult a qualified Pakistani legal professional.

## Current-law note

Pakistan's official legal sources list the **Prevention of Electronic Crimes Act, 2016** and also list the **Prevention of Electronic Crimes (Amendment) Act, 2025**. The application should therefore be treated as only as current as the PDF supplied to it.

Official sources:

- Pakistan Code — Prevention of Electronic Crimes Act, 2016
- National Assembly of Pakistan — Acts of Parliament

Always verify the latest official Gazette/legal text before relying on a legal answer.

## Troubleshooting

### Google Drive PDF download fails

Make sure the file is shared publicly:

```text
Anyone with the link
→ Viewer
```

Also verify that the file is actually a PDF.

### Groq API error

Check:

```text
GROQ_API_KEY
```

in Streamlit Secrets or the environment.

If a selected Groq model is unavailable for your account, select another available model in the sidebar.

### Out-of-memory error

The first startup can use additional RAM because the sentence-transformers model and FAISS index are loaded.

For a very large PDF:

- reduce the PDF size,
- use a smaller document,
- reduce chunk size,
- or use a lighter embedding model.

### PDF is scanned

The current version expects selectable PDF text. A scanned/image-only PDF requires OCR before embedding.

## Suggested GitHub repository name

```text
cyber-law-gpt
```

## Suggested Streamlit app title

```text
Cyber Law GPT
```
