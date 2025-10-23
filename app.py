from pinecone import Pinecone, ServerlessSpec, Vector  # Changed 'app' to 'pinecone'
import os
from dotenv import load_dotenv
import time
from pathlib import Path

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "is469"

# Create index if it doesn't exist
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1024,  # llama-text-embed-v2 dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Wait for index to be ready
while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)

index = pc.Index(index_name)

# ============================================
# DOCUMENT READING AND PROCESSING
# ============================================

def read_pdf_file(file_path):
    """Read a PDF file - requires PyPDF2"""
    try:
        import PyPDF2
        text = ""
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text += page_text
        return text
    except ImportError:
        print("PyPDF2 not installed. Install with: pip install PyPDF2")
        return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
    
    return chunks

def read_pdfs_from_folder(folder_path):
    """
    Read all PDF files from the data folder
    """
    folder = Path(folder_path)
    documents = []
    
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist!")
        return documents
    
    # Find all PDF files
    pdf_files = list(folder.glob('*.pdf'))
    
    if not pdf_files:
        print(f"No PDF files found in '{folder_path}'")
        return documents
    
    print(f"Found {len(pdf_files)} PDF file(s)")
    
    # Read each PDF
    for file_path in pdf_files:
        print(f"\nReading: {file_path.name}")
        
        content = read_pdf_file(file_path)
        
        if content:
            # Clean up the text
            content = content.strip()
            
            if not content:
                print(f"  ⚠️  Warning: {file_path.name} appears to be empty or unreadable")
                continue
            
            # Chunk the document
            chunks = chunk_text(content, chunk_size=500, overlap=50)
            print(f"  Created {len(chunks)} chunks")
            
            # Create document entries for each chunk
            for i, chunk in enumerate(chunks):
                documents.append({
                    "id": f"{file_path.stem}_chunk_{i}",
                    "text": chunk,
                    "metadata": {
                        "chunk_text": chunk,
                        "source": file_path.name,
                        "file_path": str(file_path),
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                })
    
    return documents

def embed_and_upsert_documents(documents, batch_size=100):
    """Embed documents and upsert to Pinecone in batches"""
    
    total_docs = len(documents)
    print(f"\n{'='*60}")
    print(f"Processing {total_docs} document chunks...")
    print(f"{'='*60}")
    
    for i in range(0, total_docs, batch_size):
        batch = documents[i:i + batch_size]
        vectors_to_upsert = []
        
        # Get texts for embedding
        texts = [doc["text"] for doc in batch]
        
        # Embed in batches
        print(f"\nEmbedding batch {i//batch_size + 1} ({len(batch)} chunks)...")
        try:
            embedding_response = pc.inference.embed(
                model="llama-text-embed-v2",
                inputs=texts,
                parameters={"input_type": "passage"}
            )
            
            # Prepare vectors for upsert
            for j, doc in enumerate(batch):
                vectors_to_upsert.append(
                    Vector(
                        id=doc["id"],
                        values=embedding_response[j].values,
                        metadata=doc["metadata"]
                    )
                )
            
            # Upsert to Pinecone
            index.upsert(vectors=vectors_to_upsert)
            print(f"✓ Upserted {len(vectors_to_upsert)} vectors")
            
        except Exception as e:
            print(f"✗ Error processing batch: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"✓ Successfully processed {total_docs} document chunks!")
    print(f"{'='*60}")

# ============================================
# MAIN EXECUTION
# ============================================

# Specify your data folder path
DOCUMENTS_FOLDER = "./data"  # Your PDF files are here

print(f"{'='*60}")
print(f"Reading PDF files from: {DOCUMENTS_FOLDER}")
print(f"{'='*60}")

# Read all PDFs from data folder
documents = read_pdfs_from_folder(DOCUMENTS_FOLDER)

if not documents:
    print("\n❌ No documents found or processed!")
    print("Please check:")
    print("  1. The 'data' folder exists")
    print("  2. There are PDF files in the folder")
    print("  3. The PDF files are readable")
else:
    unique_files = set(doc['metadata']['source'] for doc in documents)
    print(f"\n✓ Successfully read {len(documents)} chunks from {len(unique_files)} PDF file(s)")
    print(f"  Files: {', '.join(unique_files)}")
    
    # Embed and upload to Pinecone
    embed_and_upsert_documents(documents)
    
    # Check index stats
    stats = index.describe_index_stats()
    print(f"\nIndex '{index_name}' now contains {stats['total_vector_count']} vectors")

# ============================================
# ASSISTANT SETUP
# ============================================

assistant_name = "example-assistant"

try:
    assistant = pc.assistant.describe_assistant(assistant_name=assistant_name)
    print(f"\n✓ Using existing assistant: {assistant_name}")
except:
    assistant = pc.assistant.create_assistant(
        assistant_name=assistant_name,
        instructions="Answer in polite, short sentences. Use American English spelling and vocabulary. Base your answers on the provided context from the documents.",
        timeout=30
    )
    print(f"\n✓ Created new assistant: {assistant_name}")

# ============================================
# QUERY FUNCTION
# ============================================

def query_documents(query: str, top_k: int = 5):
    """Query the document index"""
    
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    # Embed the query
    query_embedding = pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=[query],
        parameters={"input_type": "query"}
    )
    
    # Search the index
    results = index.query(
        vector=query_embedding[0].values,
        top_k=top_k,
        include_metadata=True
    )
    
    # Display results
    print(f"\nTop {len(results['matches'])} results:\n")
    for i, match in enumerate(results['matches'], 1):
        print(f"{i}. [Score: {match['score']:.4f}] from {match['metadata']['source']}")
        print(f"   Chunk {match['metadata']['chunk_index']+1}/{match['metadata']['total_chunks']}")
        print(f"   Text preview: {match['metadata']['chunk_text'][:200]}...")
        print()
    
    return results

def chat_with_assistant(query: str):
    """Chat with the Pinecone assistant"""
    
    print(f"\n{'='*60}")
    print(f"Question: {query}")
    print(f"{'='*60}")
    
    try:
        # Create a chat with the assistant
        response = pc.assistant.chat(
            assistant_name=assistant_name,
            messages=[
                {
                    "role": "user",
                    "content": query
                }
            ]
        )
        
        print(f"\nAnswer: {response['message']['content']}")
        
        # Show citations if available
        if 'citations' in response and response['citations']:
            print(f"\nSources: {response['citations']}")
        
        return response
        
    except Exception as e:
        print(f"Error chatting with assistant: {e}")
        return None

# ============================================
# EXAMPLE QUERIES (uncomment to test)
# ============================================

if documents:  # Only query if we have documents
    print("\n" + "="*60)
    print("READY TO QUERY!")
    print("="*60)
    
    # Example 1: Direct vector search
    # query_documents("What is the main topic discussed in the documents?")
    
    # Example 2: Chat with assistant
    # chat_with_assistant("Summarize the key points from the documents")
    
    # Example 3: Specific question
    # chat_with_assistant("What are the main findings?")
    
    print("\nYou can now use:")
    print("  - query_documents('your question') for vector search")
    print("  - chat_with_assistant('your question') for AI-powered answers")