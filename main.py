import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq
from pypdf import PdfReader

app = FastAPI(title="Local RAG Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

# 1. Initialize local Sentence Transformer embedding model
# This runs locally inside Python memory—zero api limits, zero network lag.
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Initialize Groq Client for lightning fast text generation
# Automatically reads GROQ_API_KEY from your .env file
groq_client = Groq()

DB_PATH = "./my_local_vectordb"
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection("docs")


def chunk_text(text: str, chunk_size: int = 1000) -> list[str]:
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

class ChatQuery(BaseModel):
    question: str

class IngestionResponse(BaseModel):
    status: str
    chunks_found: int

@app.post("/ingest", response_model=IngestionResponse)
async def ingest_local_docs():
    if not os.path.exists("docs"):
        raise HTTPException(status_code=404, detail="local 'docs' folder not found")

    if collection.count() > 0:
        return {"status": "Database already populated", "chunks_found": collection.count()}
    
    for filename in os.listdir("docs"):
        if filename.endswith(".txt"):
            with open(f"docs/{filename}", mode="r", encoding="utf-8") as f:
                text = f.read()
            chunks = chunk_text(text)

            for i, chunk in enumerate(chunks):
                # Local encoding via sentence-transformers. 
                # .tolist() converts the numpy array to a standard list for ChromaDB
                embedding = embedding_model.encode(chunk).tolist()
                
                collection.add(
                    documents=[chunk], 
                    embeddings=[embedding], 
                    ids=[f"{filename}_chunk_{i}"],
                    metadatas=[{"source": filename, "page": None}]
                )
        if filename.endswith(".pdf"):
            reader = PdfReader(f"docs/{filename}")
            pages = reader.pages
            
            # Use enumerate to track the actual page number index safely
            for page_num, page in enumerate(pages):
                text = page.extract_text()
                if not text.strip(): # Skip pages that have no readable text
                    continue
                    
                chunks = chunk_text(text)

                for chunk_idx, chunk in enumerate(chunks):
                    embedding = embedding_model.encode(chunk).tolist()

                    # Construct a truly unique identifier across all documents
                    unique_id = f"{filename}_page_{page_num}_chunk_{chunk_idx}"

                    collection.add(
                        documents=[chunk],
                        embeddings=[embedding],
                        ids=[unique_id],
                        metadatas=[{"source": filename, "page": page_num + 1}] # Highly recommended for tracking
                    )

    return {"status": "Ingestion successful", "chunks_found": collection.count()}

@app.post("/chat")
async def chat_with_rag(query: ChatQuery):
    try:
        # Encode the incoming user query using the same local model
        query_embedding = embedding_model.encode(query.question).tolist()

        result = collection.query(query_embeddings=[query_embedding], n_results=3)

        if not result["documents"] or not result["documents"][0]:
            raise HTTPException(status_code=404, detail="No relevant content found")
    
        retrieved_chunks = result["documents"][0]
        context = "\n\n---\n\n".join(retrieved_chunks)

        system_prompt = f"""You are a helpful assistant. Answer the user's question based only on information found within the following context.
If the answer cannot be found in the context, say 'I cannot find related information in the provided docs.'

Context: 
{context}"""

        # Execute generation using Groq's high-speed endpoint
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query.question}
            ],
            temperature=0.3
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": [f"Chunk match {i+1}" for i in range(len(retrieved_chunks))]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")