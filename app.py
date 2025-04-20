import chromadb
import google.generativeai as genai
import os
import PyPDF2
from docx import Document
from chromadb.utils import embedding_functions
import uuid
import textwrap

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")   # Set your Gemini API key in environment variables
MODEL_NAME = "gemini-1.5-flash"  # Adjust based on available Gemini models
CHROMA_PATH = "chroma_db"
DOCS_DIRECTORY = "./documents"  # Directory containing your PDF and Word files
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 50  # Characters of overlap between chunks

# Initialize Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

# Initialize Chroma client
client = chromadb.PersistentClient(path=CHROMA_PATH)
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="multi-qa-mpnet-base-dot-v1")
collection = client.get_or_create_collection(name="documents", embedding_function=embedding_function)

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
            # Chunk the text
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
    Context:
    {context}
    Query: {query}
    Answer:
    """
    response = model.generate_content(prompt)
    return response.text

def rag_query(query):
    """Main RAG pipeline: retrieve and generate."""
    # Retrieve relevant document chunks and metadata
    retrieved_docs, retrieved_metadatas = retrieve_documents(query)
    # Generate response with context
    response = generate_response(query, retrieved_docs, retrieved_metadatas)
    return {
        "query": query,
        "retrieved_documents": retrieved_docs,
        "retrieved_metadatas": retrieved_metadatas,
        "response": response
    }

if __name__ == "__main__":
    # Load and chunk documents from directory
    sample_documents, metadatas = load_documents_from_directory(DOCS_DIRECTORY)

    if not sample_documents:
        print("No documents loaded. Please check the directory and file formats.")
    else:
        # Index document chunks
        index_documents(sample_documents, metadatas)

        # Example query
        query = "what is the Main AFP surveillance quality indicators?"
        result = rag_query(query)

       # print(f"Query: {result['query']}")
       # print("Retrieved Documents:")
        for doc, meta in zip(result['retrieved_documents'], result['retrieved_metadatas']):
            print(f"- [{meta['source']}, chunk {meta['chunk_index']}]: {doc[:100]}...")
        print(f"Response: {result['response']}")