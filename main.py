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
import pycld2 as cld2
import logging
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")
MODEL_NAME = "gemini-2.0-flash"
CHROMA_PATH = "chroma_db"
DOCS_DIRECTORY = "./documents"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 25
BATCH_SIZE = 100  # NEW: Batch size for indexing

# Initialize Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

# Initialize Chroma client
client = chromadb.PersistentClient(path=CHROMA_PATH)
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="multi-qa-mpnet-base-dot-v1")

# NEW: Option to reset Chroma database
def reset_chroma_db():
    """Clear the Chroma database directory."""
    try:
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
            logger.info(f"Cleared Chroma database at {CHROMA_PATH}")
    except Exception as e:
        logger.error(f"Error clearing Chroma database: {e}")

# Create or get collection
try:
    collection = client.get_or_create_collection(name="documents", embedding_function=embedding_function)
    logger.info("Chroma collection initialized")
except Exception as e:
    logger.error(f"Error initializing Chroma collection: {e}")
    raise

# FastAPI app
app = FastAPI(title="Polio and AFP Awareness Assistant API")

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
                extracted = page.extract_text() or ""
                text += extracted
            logger.info(f"Extracted text from PDF: {file_path}")
            return text.strip()
    except Exception as e:
        logger.error(f"Error reading PDF {file_path}: {e}")
        return ""

def extract_text_from_docx(file_path):
    """Extract text from a Word document."""
    try:
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        logger.info(f"Extracted text from Word: {file_path}")
        return text.strip()
    except Exception as e:
        logger.error(f"Error reading Word document {file_path}: {e}")
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
    logger.info(f"Created {len(chunks)} chunks from text")
    return chunks

def load_documents_from_directory(directory):
    """Load, extract, and chunk text from all PDF and Word files in the directory."""
    sample_documents = []
    metadatas = []
    try:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            text = ""
            if filename.lower().endswith(".pdf"):
                text = extract_text_from_pdf(file_path)
            elif filename.lower().endswith((".docx", ".doc")):
                text = extract_text_from_docx(file_path)

            if text:
                chunks = chunk_text(text)
                for i, chunk in enumerate(chunks):
                    sample_documents.append(chunk)
                    metadatas.append({"source": filename, "chunk_index": i})
                logger.info(f"Processed {len(chunks)} chunks from {filename}")
    except Exception as e:
        logger.error(f"Error loading documents from {directory}: {e}")
    logger.info(f"Loaded {len(sample_documents)} total chunks from directory")
    return sample_documents, metadatas

def index_documents(documents, metadatas):
    """Index document chunks into Chroma vector store with metadata in batches."""
    try:
        total_chunks = len(documents)
        logger.info(f"Starting to index {total_chunks} chunks")
        for i in range(0, total_chunks, BATCH_SIZE):
            batch_docs = documents[i:i + BATCH_SIZE]
            batch_metas = metadatas[i:i + BATCH_SIZE]
            batch_ids = [str(uuid.uuid4()) for _ in batch_docs]
            collection.add(
                documents=batch_docs,
                ids=batch_ids,
                metadatas=batch_metas
            )
            logger.info(f"Indexed batch {i // BATCH_SIZE + 1}: {len(batch_docs)} chunks")
        logger.info(f"Completed indexing {total_chunks} chunks")
    except Exception as e:
        logger.error(f"Error indexing documents: {e}")
        raise

def detect_language(query):
    """Detect the language of the query using pycld2."""
    try:
        _, _, details = cld2.detect(query)
        lang_code = details[0][1]
        lang_map = {
            'am': 'Amharic',
            'en': 'English',
            'om': 'Afaan Oromo'
        }
        logger.info(f"Detected language: {lang_map.get(lang_code, 'English')} for query: {query[:50]}")
        return lang_map.get(lang_code, 'English')
    except Exception as e:
        logger.error(f"Error detecting language: {e}")
        return 'English'  # Default fallback

def retrieve_documents(query, n_results=7):
    """Retrieve relevant document chunks from Chroma based on query."""
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas"]
        )
        logger.info(f"Retrieved {len(results['documents'][0])} documents for query: {query[:50]}")
        return results['documents'][0], results['metadatas'][0]
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        raise

def generate_response(query, context_docs, metadatas):
    """Generate a response using Gemini API with retrieved context in the detected language."""
    detected_language = detect_language(query)
    context = "\n".join([f"[{meta['source']}, chunk {meta['chunk_index']}]: {doc}" for doc, meta in zip(context_docs, metadatas)])
    prompt = f"""
You are a Polio and Acute Flaccid Paralysis (AFP) Awareness Assistant Chatbot.
Your purpose is to provide clear, accurate, and helpful information to raise awareness about polio and AFP.
You can assist with topics including:
- Symptoms and causes of polio and AFP
- Prevention methods, including vaccination
- Treatment options and supportive care
- Importance of early detection and reporting
- General health advice related to polio prevention
Your responses must be:
- Professional, polite, and supportive.
- Concise yet comprehensive, aiming to fully resolve the user's query.
- In the language: {detected_language} (respond only in Amharic, English, or Afaan Oromo as detected).
If you are asked a question outside your scope or lack enough information, politely inform the user and guide them to contact local health authorities, the World Health Organization (WHO), or a healthcare provider for further assistance.
Your goal is to educate users about polio and AFP, promote prevention, and encourage timely medical action in a user-friendly manner.
Use the following context to answer the question.
Always respond as a fully knowledgeable and confident assistant.
Summarize and make it modern like you're responding to a public health query, without referencing where the information comes from.
If specific details are missing, do not express uncertainty or suggest a lack of information.
Instead: Provide the most complete, authoritative response possible based on general knowledge, best practices, and reasonable assumptions about polio and AFP.
Respond using the following language only: {detected_language}.
Frame every answer as if it is coming from your own expertise as a polio awareness assistant.
Avoid language like “I don’t have enough information” or “the document doesn’t say”; instead, use confident, helpful statements that guide and reassure the user.
If possible, guide them to local health authorities or WHO resources.
Your goal is to make the user feel fully supported and informed, without gaps or signs of limitation.
If you don’t have specific information, provide a general answer based on standard public health practices and suggest contacting a healthcare provider for precise details.
Context:
{context}
Query: {query}
Answer:
"""
    try:
        response = model.generate_content(prompt)
        logger.info(f"Generated response for query: {query[:50]}")
        return response.text
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise

# Load and index documents at startup
logger.info("Starting document loading")
sample_documents, metadatas = load_documents_from_directory(DOCS_DIRECTORY)
if sample_documents:
    logger.info("Documents loaded, starting indexing")
    index_documents(sample_documents, metadatas)
else:
    logger.warning("No documents loaded. Please check the directory and file formats.")

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
            response=response
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/")
async def serve_ui():
    """Serve the UI."""
    return StaticFiles(directory="static", html=True)

if __name__ == "__main__":
    import uvicorn
    # Uncomment the following line to reset Chroma database on startup (use with caution)
    # reset_chroma_db()
    uvicorn.run(app, host="0.0.0.0", port=8000)
