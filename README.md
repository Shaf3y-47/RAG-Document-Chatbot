Markdown
# RAG Document Chatbot

A high-performance Retrieval-Augmented Generation (RAG) system that allows users to ingest local `.txt` and `.pdf` files and query them using natural language. Built with a FastAPI backend for modular processing and a Streamlit interface for an interactive user experience.

## Features
- **File Processing**: Native handling of `.txt` and `.pdf` files.
- **Metadata Awareness**: Ingests files with metadata (filename, size, page number) for accurate retrieval.
- **Semantic Search**: Vector-based search using `bge-small-en-v1.5` embeddings for highly accurate document retrieval.
- **AI Integration**: Context-aware answers powered by Llama 3 via the Groq API.
- **Database Management**: Built-in state reset functionality to instantly clear the vector database and chat history for fresh queries.

## Tech Stack
- **Backend**: FastAPI (Python)
- **Frontend**: Streamlit
- **Vector DB**: ChromaDB (Cloud storage)
- **Embeddings**: SentenceTransformers
- **LLM**: Groq API (Llama-3.3-70b)

## Setup & Installation

1. **Clone the repository:**
```bash
   git clone https://github.com/Shaf3y-47/RAG-Document-Chatbot.git
   cd RAG-Document-Chatbot
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```
Install dependencies:

```bash
pip install -r requirements.txt
```
Environment Configuration:
Create a `.env` file in the root directory and add your API keys and database credentials:

```text
GROQ_API_KEY=your_groq_api_key
CHROMA_KEY=your_chroma_cloud_api_key
CHROMA_TENANT=your_chroma_tenant_name
CHROMA_NAME=your_chroma_database_name
```
---
## Running the Application
This project requires two terminals to run simultaneously:

**Terminal 1: Start the FastAPI Backend**

```bash
uvicorn main:app --reload
```
**Terminal 2: Start the Streamlit Frontend**

```bash
streamlit run app.py
```