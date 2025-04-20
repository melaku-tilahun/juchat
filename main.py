import chromadb
import google.generativeai as genai
import os
import PyPDF2
from docx import Document
from chromadb.utils import embedding_functions
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any


# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")
MODEL_NAME = "gemini-1.5-flash"
CHROMA_PATH = "chroma_db"
DOCS_DIRECTORY = "./documents"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 50

# Initialize Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

# Initialize Chroma client
client = chromadb.PersistentClient(path=CHROMA_PATH)
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="multi-qa-mpnet-base-dot-v1")
collection = client.get_or_create_collection(name="documents", embedding_function=embedding_function)

# FastAPI app
app = FastAPI(title="Polio (AFP) Awareness RAG API")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic model for request body
class QueryRequest(BaseModel):
    query: str

# Pydantic model for response
class QueryResponse(BaseModel):
    query: str
    retrieved_documents: List[str]
    retrieved_metadatas: List[Dict[str, Any]]
    response: str

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text.strip()
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        return ""

def extract_text_from_docx(file_path):
    """Extract text from a Word document."""
    try:
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        return text.strip()
    except Exception as e:
        print(f"Error reading Word document {file_path}: {e}")
        return ""

def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split text into chunks with specified size and overlap."""
    if not text:
        return []
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

def load_documents_from_directory(directory):
    """Load, extract, and chunk text from all PDF and Word files in the directory."""
    sample_documents = []
    metadatas = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        text = ""
        if filename.lower().endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
            if text:
                print(f"Loaded PDF: {filename}")
        elif filename.lower().endswith((".docx", ".doc")):
            text = extract_text_from_docx(file_path)
            if text:
                print(f"Loaded Word: {filename}")

        if text:
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                sample_documents.append(chunk)
                metadatas.append({"source": filename, "chunk_index": i})

    return sample_documents, metadatas

def index_documents(documents, metadatas):
    """Index document chunks into Chroma vector store with metadata."""
    doc_ids = [str(uuid.uuid4()) for _ in documents]
    collection.add(
        documents=documents,
        ids=doc_ids,
        metadatas=metadatas
    )
    print(f"Indexed {len(documents)} chunks.")

def retrieve_documents(query, n_results=7):
    """Retrieve relevant document chunks from Chroma based on query."""
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas"]
    )
    return results['documents'][0], results['metadatas'][0]

def generate_response(query, context_docs, metadatas):
    """Generate a response using Gemini API with retrieved context."""
    context = "\n".join([f"[{meta['source']}, chunk {meta['chunk_index']}]: {doc}" for doc, meta in zip(context_docs, metadatas)])
    prompt = f"""
You are a knowledgeable and supportive assistant dedicated to raising awareness about Polio (AFP).
Your goal is to provide clear, accurate, and easy-to-understand answers to user questions.
Focus on promoting understanding of Polio and AFP, encouraging prevention through vaccination and hygiene, 
correcting misinformation with facts, and responding with empathy and cultural sensitivity.
Always use simple language while maintaining medical accuracy to help users take informed health actions.
respond to user queries only using the context provided.
    Context:
    {context}
    Query: {query}
    Answer:
    """
    response = model.generate_content(prompt)
    return response.text

# Load and index documents at startup
sample_documents, metadatas = load_documents_from_directory(DOCS_DIRECTORY)
if sample_documents:
    index_documents(sample_documents, metadatas)
else:
    print("No documents loaded. Please check the directory and file formats.")

@app.post("/rag_query", response_model=QueryResponse)
async def rag_query_endpoint(request: QueryRequest):
    """Handle RAG query and return response with retrieved documents."""
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        retrieved_docs, retrieved_metadatas = retrieve_documents(query)
        response = generate_response(query, retrieved_docs, retrieved_metadatas)
        return QueryResponse(
            query=query,
            retrieved_documents=retrieved_docs,
            retrieved_metadatas=retrieved_metadatas,
            response= response
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/")
async def serve_ui():
    """Serve the UI."""
    return StaticFiles(directory="static", html=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)