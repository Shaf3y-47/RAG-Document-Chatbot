# RAG Document Chatbot

A high-performance Retrieval-Augmented Generation (RAG) system that allows users to ingest local `.txt` and `.pdf` files and query them using natural language. Built with a FastAPI backend for modular processing and a Streamlit interface for an interactive user experience.

## Features
- **File Processing**: Native handling of `.txt` and `.pdf` files.
- **Metadata Awareness**: Ingests files with metadata (filename, size, page number) for accurate retrieval.
- **Semantic Search**: Vector-based search using `all-MiniLM-L6-v2` embeddings.
- **AI Integration**: Context-aware answers powered by Llama 3 via Groq API.

## Tech Stack
- **Backend**: FastAPI (Python)
- **Frontend**: Streamlit
- **Vector DB**: ChromaDB (Persistent local storage)
- **Embeddings**: SentenceTransformers
- **LLM**: Groq API (Llama-3.3-70b)

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Shaf3y-47/RAG-Document-Chatbot
   cd "RAG-Document-Chatbot"

Create and activate a virtual environment:

Bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

Install dependencies:

Bash
pip install -r requirements.txt

Environment Configuration:
Create a .env file in the root directory and add your Groq API key:
-> GROQ_API_KEY=your_actual_api_key_here

Running the Application
This project requires two terminals to run:

-> Start the FastAPI Backend:

Bash
uvicorn main:app --reload

-> Start the Streamlit Frontend:

Bash
streamlit run app.py