import os
import io
from fastapi import FastAPI, HTTPException, UploadFile
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

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

DB_PATH = "./my_local_vectordb"
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection("docs")


def chunk_text(text: str, chunk_size: int = 500, overlap= 50) -> list[str]:
    start = 0
    chunks= []
    while start < len(text):
        end= start+chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    
    return chunks

class ChatQuery(BaseModel):
    question: str

@app.post("/ingest")
def Upload(file: UploadFile | None= None):
    if not file:
        return {"message":"Nothing uploaded."}
    else:
        try:
            filename = file.filename
            inputs = file.file.read()
            if filename.endswith(".txt"):
                result = inputs.decode("utf-8")
                chunks = chunk_text(result)
                doc_metadata = {
                "filename": file.filename,
                "content_type": file.content_type,
                "size_bytes": len(inputs)
            }
                for i, chunk in enumerate(chunks):
                    embedding = embedding_model.encode(chunk).tolist()
                            
                    collection.add(
                        documents=[chunk], 
                        embeddings=[embedding], 
                        ids=[f"{filename}_chunk_{i}"],
                        metadatas=[doc_metadata]
                    )
            if filename.endswith(".pdf"):
                reader = PdfReader(io.BytesIO(inputs))
                pages = reader.pages
                        
                for page_num, page in enumerate(pages):
                    text = page.extract_text()
                    if not text.strip():
                        continue
                                
                    chunks = chunk_text(text)

                    for chunk_idx, chunk in enumerate(chunks):
                        embedding = embedding_model.encode(chunk).tolist()

                        unique_id = f"{filename}_page_{page_num}_chunk_{chunk_idx}"

                        collection.add(
                            documents=[chunk],
                            embeddings=[embedding],
                            ids=[unique_id],
                            metadatas=[{"source": filename, "page": page_num + 1}]
                        )
        except Exception as e:
            return {"Erorr details":f"Unsuccessful upload error detail{e}."}


    return {"status": "Ingestion successful", "chunks_found": collection.count()}


@app.post("/chat")
def chat_with_rag(query: ChatQuery):
    try:
        query_embedding = embedding_model.encode(query.question).tolist()

        result = collection.query(query_embeddings=[query_embedding], n_results=3)

        if not result["documents"] or not result["documents"][0]:
            raise HTTPException(status_code=404, detail="No relevant content found")
    
        retrieved_chunks = result["documents"][0]
        context = "\n\n---\n\n".join(retrieved_chunks)

        system_prompt = f"""
        You are a helpful assistant. Answer the user's question based only on information found within the following context.
        If the answer cannot be found in the context, say 'I cannot find related information in the provided docs.'
        Context: 
        {context}
        """

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