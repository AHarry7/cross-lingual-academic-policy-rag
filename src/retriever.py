import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.document_processor import load_and_chunk_documents


DB_DIR = "vector_db"

def get_embedding_model():
    # The multilingual brain that bridges English and Roman Urdu
    print("Loading multilingual embedding model...")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def build_or_load_vector_store():
    embeddings = get_embedding_model()
    
    # Check if database already exists so we don't rebuild it every time
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        print("Loading existing Vector Database from local storage...")
        vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    else:
        print("Building new Vector Database. This might take a minute...")
        chunks = load_and_chunk_documents()
        
        if not chunks:
            raise ValueError("No documents found to build the database.")
        
        vector_store = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings, 
            persist_directory=DB_DIR
        )
        print("Database built and saved successfully!")
        
    return vector_store

# This block allows us to test the cross-lingual search instantly
if __name__ == "__main__":
    db = build_or_load_vector_store()
    
    # IMPORTANT: Change this test query to something relevant to your specific PDFs!
    test_query = "grading policy kia hai?" 
    
    print(f"\n--- TEST: Searching for Roman Urdu query: '{test_query}' ---")
    
    # Fetch the top 2 most mathematically similar English chunks
    results = db.similarity_search(test_query, k=2)
    
    for i, res in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(res.page_content[:400] + "...\n")