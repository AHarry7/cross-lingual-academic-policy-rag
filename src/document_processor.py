from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_documents(data_path="data/"):
    print(f"Scanning the '{data_path}' directory for PDFs...")
    loader = PyPDFDirectoryLoader(data_path)
    documents = loader.load()
    
    if not documents:
        print("No PDFs found! Please add some to the data folder.")
        return []
        
    print(f"Successfully loaded {len(documents)} pages.")

    # We split the text into chunks so the Vector DB can process it efficiently
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200 # Overlap prevents cutting a sentence in half
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Divided the pages into {len(chunks)} manageable chunks.")
    
    return chunks

# This block allows us to test this specific file directly
if __name__ == "__main__":
    test_chunks = load_and_chunk_documents()
    if test_chunks:
        print("\n--- TEST: Here is a preview of the first chunk ---")
        print(test_chunks[0].page_content[:300] + "...")