# RAG Document Chatbot

A retrieval-augmented generation (RAG) application that lets you ask questions about your documents.

## Features
- Ingest PDFs and text files
- Semantic search using embeddings
- AI-powered answers via Groq's Llama API
- Clean Streamlit UI

## Tech Stack
- Streamlit (frontend)
- FastAPI (backend)
- ChromaDB (vector database)
- Sentence Transformers (embeddings)
- Groq API (LLM)

## Local Setup
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running
```bash
# Terminal 1: FastAPI backend
uvicorn your_fastapi_file:app --reload

# Terminal 2: Streamlit frontend
streamlit run app.py
```