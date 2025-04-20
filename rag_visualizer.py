import chromadb
import os
import PyPDF2
from docx import Document
from chromadb.utils import embedding_functions
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import uuid

# Configuration
CHROMA_PATH = "chroma_db"  # Directory where ChromaDB stores vector data
DOCS_DIRECTORY = "./documents"  # Directory containing your Word/PDF files
CHUNK_SIZE = 1000  # Maximum characters per text chunk
CHUNK_OVERLAP = 50  # Overlap between chunks to maintain context

# Initialize Chroma client and embedding function
client = chromadb.PersistentClient(path=CHROMA_PATH)
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="multi-qa-mpnet-base-dot-v1")
try:
    # Reset the collection to avoid duplicate folders (comment out to keep existing data)
    try:
        client.delete_collection(name="documents")
        print("Deleted existing collection to start fresh.")
    except:
        pass  # Collection may not exist, which is fine
    collection = client.get_or_create_collection(name="documents", embedding_function=embedding_function)
except Exception as e:
    print(f"Error initializing collection: {e}")
    raise

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
    chunk_lengths = []
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
            chunk_lengths.extend([len(chunk) for chunk in chunks])
            for i, chunk in enumerate(chunks):
                sample_documents.append(chunk)
                metadatas.append({"source": filename, "chunk_index": i})

    return sample_documents, metadatas, chunk_lengths

def index_documents(documents, metadatas):
    """Index document chunks into Chroma vector store with metadata."""
    try:
        doc_ids = [str(uuid.uuid4()) for _ in documents]
        collection.add(
            documents=documents,
            ids=doc_ids,
            metadatas=metadatas
        )
        print(f"Indexed {len(documents)} chunks.")
    except Exception as e:
        print(f"Error indexing documents: {e}")
        raise

def get_embeddings_and_metadata():
    """Retrieve embeddings and metadata from ChromaDB."""
    try:
        results = collection.get(include=["documents", "metadatas", "embeddings"])
        return results["embeddings"], results["metadatas"], results["documents"]
    except Exception as e:
        print(f"Error retrieving embeddings: {e}")
        return [], [], []

def visualize_embeddings(embeddings, metadatas):
    """
    Visualize document chunk embeddings using t-SNE.
    - Each point represents a text chunk from your documents.
    - Points are colored by source document to show which chunks belong to which file.
    - Clusters of points indicate similar content (e.g., chunks about similar topics).
    - Helps understand how well the RAG system groups related information.
    """
    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings)
    
    # Apply t-SNE to reduce 768-dimensional embeddings to 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings_array)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        "x": embeddings_2d[:, 0],
        "y": embeddings_2d[:, 1],
        "source": [meta["source"] for meta in metadatas]
    })
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=df, x="x", y="y", hue="source", style="source", s=150, alpha=0.7)
    plt.title("t-SNE Visualization of Document Chunk Embeddings\n(Shows how text chunks are grouped by content similarity)", fontsize=14, pad=20)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Source Document", fontsize=10)
    
    # Add explanation text box
    explanation = (
        "What this shows:\n"
        "- Each dot is a text chunk from your documents.\n"
        "- Colors represent different source files.\n"
        "- Dots close together have similar content (e.g., same topic).\n"
        "- This helps check if the RAG system understands content relationships."
    )
    plt.text(0.02, 0.98, explanation, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig("embeddings_tsne.png", bbox_inches="tight", dpi=300)
    plt.close()

def visualize_similarity(query, n_results=7):
    """
    Visualize cosine similarity between a query and retrieved document chunks.
    - Bars show how closely each chunk matches the query (higher = more relevant).
    - Helps evaluate if the RAG system retrieves relevant chunks for a question.
    - Query example: 'What are the main AFP surveillance quality indicators?'
    """
    # Get query embedding
    query_embedding = embedding_function([query])[0]
    
    # Retrieve documents
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "embeddings"]
    )
    doc_embeddings = np.array(results["embeddings"][0])
    metadatas = results["metadatas"][0]
    
    # Compute cosine similarities
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        "Document": [f"{meta['source']} (Chunk {meta['chunk_index']})" for meta in metadatas],
        "Similarity": similarities
    })
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(data=df, x="Similarity", y="Document", palette="Blues_d")
    plt.title(f"Cosine Similarity of Query: '{query}'\n(Shows how relevant each chunk is to the question)", fontsize=14, pad=20)
    plt.xlabel("Cosine Similarity (0 to 1, higher = more relevant)", fontsize=12)
    plt.ylabel("Document Chunk", fontsize=12)
    
    # Add explanation text box
    explanation = (
        "What this shows:\n"
        "- Each bar is a retrieved text chunk.\n"
        "- Bar length shows how similar the chunk is to the query.\n"
        "- Longer bars mean the chunk is more relevant to the question.\n"
        "- This checks if the RAG system finds the right information."
    )
    plt.text(0.02, 0.98, explanation, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig("similarity_plot.png", bbox_inches="tight", dpi=300)
    plt.close()

def visualize_chunk_sizes(chunk_lengths):
    """
    Visualize the distribution of chunk sizes.
    - Shows how long each text chunk is (in characters).
    - Most chunks should be close to 1000 characters (CHUNK_SIZE).
    - Helps verify if documents are split evenly for the RAG system.
    """
    plt.figure(figsize=(10, 8))
    sns.histplot(chunk_lengths, bins=30, kde=True, color='skyblue')
    plt.title("Distribution of Document Chunk Sizes\n(Shows how text is split for RAG processing)", fontsize=14, pad=20)
    plt.xlabel("Chunk Length (Characters)", fontsize=12)
    plt.ylabel("Number of Chunks", fontsize=12)
    
    # Add explanation text box
    explanation = (
        "What this shows:\n"
        "- Each bar shows how many chunks have a certain length.\n"
        "- Most chunks should be around 1000 characters.\n"
        "- Smaller chunks are from document ends or short sections.\n"
        "- This checks if text is split consistently for the RAG system."
    )
    plt.text(0.02, 0.98, explanation, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig("chunk_sizes.png", bbox_inches="tight", dpi=300)
    plt.close()

def main():
    # Load documents from the ./documents directory
    sample_documents, metadatas, chunk_lengths = load_documents_from_directory(DOCS_DIRECTORY)
    
    if not sample_documents:
        print("No documents loaded. Please check the './documents' directory and ensure it contains valid PDF or Word files.")
        return
    
    # Index documents into ChromaDB for vector search
    index_documents(sample_documents, metadatas)
    
    # Retrieve embeddings (numerical representations of text chunks)
    embeddings, stored_metadatas, _ = get_embeddings_and_metadata()
    
    # Debug: Print embeddings info to verify data
    embeddings_length = len(embeddings) if embeddings is not None else 0
    print(f"Embeddings type: {type(embeddings)}, length: {embeddings_length}")
    
    if embeddings is None or embeddings_length == 0:
        print("No embeddings found in ChromaDB. Indexing may have failed.")
        return
    
    # Generate visualizations with explanations
    visualize_embeddings(embeddings, stored_metadatas)
    visualize_similarity("What are the main AFP surveillance quality indicators?")
    visualize_chunk_sizes(chunk_lengths)
    print("Visualizations saved as PNG files: embeddings_tsne.png, similarity_plot.png, chunk_sizes.png")

if __name__ == "__main__":
    main()