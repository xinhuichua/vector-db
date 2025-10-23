import streamlit as st
from pinecone import Pinecone, ServerlessSpec, Vector
import os
from dotenv import load_dotenv
import time
from pathlib import Path
import PyPDF2

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

# Initialize Pinecone (cached to avoid recreating connection)
@st.cache_resource
def init_pinecone():
    """Initialize Pinecone connection"""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    return pc

@st.cache_resource
def get_index(_pc, index_name="is469"):
    """Get or create Pinecone index"""
    # Create index if it doesn't exist
    if not _pc.has_index(index_name):
        _pc.create_index(
            name=index_name,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        # Wait for index to be ready
        while not _pc.describe_index(index_name).status['ready']:
            time.sleep(1)
    
    return _pc.Index(index_name)

# PDF Processing Functions
def read_pdf_file(file_path):
    """Read a PDF file"""
    try:
        text = ""
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading {file_path}: {e}")
        return None

def read_uploaded_pdf(uploaded_file):
    """Read an uploaded PDF file from Streamlit"""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def process_pdf(uploaded_file, pc, index):
    """Process a single uploaded PDF"""
    content = read_uploaded_pdf(uploaded_file)
    
    if not content or not content.strip():
        return None
    
    # Chunk the document
    chunks = chunk_text(content, chunk_size=500, overlap=50)
    
    documents = []
    file_stem = Path(uploaded_file.name).stem
    
    for i, chunk in enumerate(chunks):
        documents.append({
            "id": f"{file_stem}_chunk_{i}",
            "text": chunk,
            "metadata": {
                "chunk_text": chunk,
                "source": uploaded_file.name,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
        })
    
    return documents

def embed_and_upsert_documents(documents, pc, index, batch_size=100):
    """Embed documents and upsert to Pinecone"""
    total_docs = len(documents)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, total_docs, batch_size):
        batch = documents[i:i + batch_size]
        texts = [doc["text"] for doc in batch]
        
        status_text.text(f"Embedding batch {i//batch_size + 1}...")
        
        try:
            embedding_response = pc.inference.embed(
                model="llama-text-embed-v2",
                inputs=texts,
                parameters={"input_type": "passage"}
            )
            
            vectors_to_upsert = []
            for j, doc in enumerate(batch):
                vectors_to_upsert.append(
                    Vector(
                        id=doc["id"],
                        values=embedding_response[j].values,
                        metadata=doc["metadata"]
                    )
                )
            
            index.upsert(vectors=vectors_to_upsert)
            progress_bar.progress((i + len(batch)) / total_docs)
            
        except Exception as e:
            st.error(f"Error processing batch: {e}")
            continue
    
    status_text.text("✓ All documents processed!")
    progress_bar.progress(1.0)

def query_documents(query, pc, index, top_k=5):
    """Query the document index"""
    query_embedding = pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=[query],
        parameters={"input_type": "query"}
    )
    
    results = index.query(
        vector=query_embedding[0].values,
        top_k=top_k,
        include_metadata=True
    )
    
    return results

def chat_with_assistant(query, pc, index, top_k=5):
    """Query using vector search and generate response"""
    try:
        # Get relevant context using vector search
        results = query_documents(query, pc, index, top_k)
        
        if not results['matches']:
            return {
                'message': {'content': "I couldn't find any relevant information in the documents to answer your question."},
                'citations': []
            }
        
        # Combine context from top results
        context_parts = []
        citations = []
        
        for i, match in enumerate(results['matches']):
            context_parts.append(match['metadata']['chunk_text'])
            citations.append({
                'source': match['metadata']['source'],
                'score': match['score'],
                'chunk': f"{match['metadata']['chunk_index']+1}/{match['metadata']['total_chunks']}"
            })
        
        context = "\n\n".join(context_parts)
        
        # Create a simple response based on context
        response_text = f"Based on the documents, here's what I found:\n\n{context[:1000]}..."
        
        return {
            'message': {'content': response_text},
            'citations': citations
        }
    except Exception as e:
        st.error(f"Error processing query: {e}")
        return None

# Main Streamlit App
def main():
    st.title("📚 PDF RAG Assistant")
    st.markdown("Upload PDFs and ask questions about their content using AI-powered search")
    
    # Initialize connections
    try:
        pc = init_pinecone()
        index = get_index(pc)
    except Exception as e:
        st.error(f"Error initializing Pinecone: {e}")
        st.stop()
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📁 Document Management")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=['pdf'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("Process Uploaded PDFs", type="primary"):
                all_documents = []
                
                with st.spinner("Reading PDFs..."):
                    for uploaded_file in uploaded_files:
                        st.write(f"Processing: {uploaded_file.name}")
                        docs = process_pdf(uploaded_file, pc, index)
                        if docs:
                            all_documents.extend(docs)
                
                if all_documents:
                    st.write(f"Total chunks: {len(all_documents)}")
                    with st.spinner("Embedding and uploading to Pinecone..."):
                        embed_and_upsert_documents(all_documents, pc, index)
                    st.success(f"✓ Processed {len(all_documents)} chunks from {len(uploaded_files)} file(s)!")
                else:
                    st.error("No documents were processed successfully")
        
        st.divider()
        
        # Index stats
        st.header("📊 Index Statistics")
        if st.button("Refresh Stats"):
            stats = index.describe_index_stats()
            st.metric("Total Vectors", stats.get('total_vector_count', 0))
    
    # Main chat interface
    st.header("💬 Ask Questions")
    
    # Create tabs for different query methods
    tab1, tab2 = st.tabs(["🤖 AI Assistant", "🔍 Vector Search"])
    
    with tab1:
        st.markdown("Ask questions and get AI-powered answers based on your documents")
        
        # Chat history
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chat_with_assistant(prompt, pc, index)
                    
                    if response:
                        answer = response['message']['content']
                        st.markdown(answer)
                        
                        # Show citations if available
                        if 'citations' in response and response['citations']:
                            with st.expander("📚 Sources"):
                                for i, citation in enumerate(response['citations'], 1):
                                    st.write(f"**{i}. {citation['source']}** (Score: {citation['score']:.4f}, Chunk: {citation['chunk']})")
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        st.error("Failed to get response from assistant")
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    with tab2:
        st.markdown("Direct vector search to find relevant document chunks")
        
        query = st.text_input("Enter your search query:", key="vector_search")
        top_k = st.slider("Number of results:", 1, 10, 5)
        
        if st.button("Search", type="primary"):
            if query:
                with st.spinner("Searching..."):
                    results = query_documents(query, pc, index, top_k)
                    
                    if results['matches']:
                        st.success(f"Found {len(results['matches'])} results")
                        
                        for i, match in enumerate(results['matches'], 1):
                            with st.expander(f"Result {i} - Score: {match['score']:.4f} - {match['metadata']['source']}"):
                                st.markdown(f"**Source:** {match['metadata']['source']}")
                                st.markdown(f"**Chunk:** {match['metadata']['chunk_index']+1}/{match['metadata']['total_chunks']}")
                                st.markdown("**Content:**")
                                st.write(match['metadata']['chunk_text'])
                    else:
                        st.info("No results found")
            else:
                st.warning("Please enter a search query")

if __name__ == "__main__":
    main()