import os
import chromadb
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- CONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the consistent paths and names for the database
DB_DIRECTORY = os.path.join(script_dir, "render_disk/chroma_db")
RESOURCE_DIRECTORY = os.path.join(script_dir, "resource")
COLLECTION_NAME = "medical_manuals"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 50

# --- MAIN EXECUTION ---
def main():
    """Main function to build the ChromaDB vector database from PDF documents."""
    print("--- Starting Database Build Process ---")

    # Initialize the ChromaDB client, persisting to the specified directory
    client = chromadb.PersistentClient(path=DB_DIRECTORY)

    # Check if collection exists and delete it for a fresh start
    existing_collections = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_collections:
        print(f"Collection '{COLLECTION_NAME}' already exists. Deleting it for a fresh build.")
        client.delete_collection(name=COLLECTION_NAME)

    # Create a new collection. By not specifying an embedding function,
    # ChromaDB will use its default: all-MiniLM-L6-v2 (dimension 384)
    print(f"Creating new collection: '{COLLECTION_NAME}'")
    collection = client.create_collection(name=COLLECTION_NAME)

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    # Ensure the resource directory exists
    if not os.path.exists(RESOURCE_DIRECTORY):
        print(f"Error: Resource directory not found at '{RESOURCE_DIRECTORY}'")
        print("Please create the 'resource' folder and place your PDF files inside it.")
        return

    # Process each PDF file in the resource directory
    pdf_files = [f for f in os.listdir(RESOURCE_DIRECTORY) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in '{RESOURCE_DIRECTORY}'. Nothing to process.")
        return

    print(f"Found {len(pdf_files)} PDF(s) to process...")

    for pdf_file in pdf_files:
        try:
            pdf_path = os.path.join(RESOURCE_DIRECTORY, pdf_file)
            print(f"\n--- Processing '{pdf_file}' ---")
            
            # Read the PDF content
            reader = PdfReader(pdf_path)
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() or ""
            
            if not full_text.strip():
                print(f"Warning: No text could be extracted from '{pdf_file}'. Skipping.")
                continue

            # Split the text into chunks
            print("Splitting document into text chunks...")
            chunks = text_splitter.split_text(full_text)
            
            total_chunks = len(chunks)
            print(f"Document split into {total_chunks} chunks. Adding to database...")
            
            # Add chunks to the collection in batches to be efficient
            batch_size = 100 
            for i in range(0, total_chunks, batch_size):
                batch_chunks = chunks[i:i + batch_size]
                
                # Create unique IDs for each chunk
                batch_ids = [f"{pdf_file}-chunk-{i+j}" for j in range(len(batch_chunks))]
                
                collection.add(
                    documents=batch_chunks,
                    ids=batch_ids
                )
                print(f"  > Added batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}...")

        except Exception as e:
            print(f"An error occurred while processing {pdf_file}: {e}")
            continue

    print("\n--- Database build process complete! ---")
    print(f"Total documents in collection '{COLLECTION_NAME}': {collection.count()}")

if __name__ == "__main__":
    main()