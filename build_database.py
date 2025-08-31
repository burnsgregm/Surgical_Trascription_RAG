import os
import time
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
import chromadb

# --- Configuration ---
# Make the code more robust by loading .env from the script's directory
script_dir = os.path.dirname(os.path.realpath(__file__))
dotenv_path = os.path.join(script_dir, '.env')
load_dotenv(dotenv_path=dotenv_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it as an environment variable.")

genai.configure(api_key=GEMINI_API_KEY)

RESOURCE_DIR = "resource"
# Save DB to a path compatible with Render's persistent disks
# This also works well for local testing.
DB_PATH = os.path.join(script_dir, "render_disk", "chroma_db")
COLLECTION_NAME = "merck_manual"


def get_pdf_texts(pdf_docs_paths):
    """Extracts text from a list of PDF file paths."""
    full_text = ""
    for pdf_path in pdf_docs_paths:
        try:
            pdf_reader = PdfReader(pdf_path)
            for page in pdf_reader.pages:
                full_text += page.extract_text() or ""
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
    return full_text

def get_text_chunks(text):
    """Splits text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,  # <-- Chunk size updated to 1024
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def embed_and_store_chunks(chunks, collection):
    """Embeds text chunks and stores them in ChromaDB, handling rate limits."""
    batch_size = 100  # Process chunks in batches to be safe with API limits
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        
        # Create unique IDs for this batch
        batch_ids = [f"chunk_{i+j}" for j in range(len(batch_chunks))]
        
        print(f"Embedding batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}...")
        
        try:
            # Embed the batch of chunks
            response = genai.embed_content(
                model="models/embedding-001",
                content=batch_chunks,
                task_type="retrieval_document"
            )
            embeddings = response['embedding']

            # Store the embeddings and documents in ChromaDB
            collection.add(
                embeddings=embeddings,
                documents=batch_chunks,
                ids=batch_ids
            )
            
        except Exception as e:
            print(f"An error occurred during embedding or storage: {e}")
            print("Retrying after a short delay...")
            time.sleep(5) # Wait before retrying
            # Simple retry logic, can be made more robust
            try:
                response = genai.embed_content(
                    model="models/embedding-001",
                    content=batch_chunks,
                    task_type="retrieval_document"
                )
                embeddings = response['embedding']
                collection.add(embeddings=embeddings, documents=batch_chunks, ids=batch_ids)
            except Exception as final_e:
                print(f"Retry failed for batch starting at chunk {i}. Error: {final_e}")
                continue # Skip to the next batch

def main():
    """Main function to build the ChromaDB database."""
    print("Starting database build process...")

    # 1. Setup ChromaDB client and collection
    if not os.path.exists(DB_PATH):
        os.makedirs(DB_PATH)
        print(f"Created database directory at {DB_PATH}")

    client = chromadb.PersistentClient(path=DB_PATH)
    
    # Check if collection exists and delete it for a fresh start
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        print(f"Collection '{COLLECTION_NAME}' already exists. Deleting it for a fresh build.")
        client.delete_collection(name=COLLECTION_NAME)

    collection = client.create_collection(name=COLLECTION_NAME)
    print(f"ChromaDB collection '{COLLECTION_NAME}' created successfully.")

    # 2. Find and process PDF files one by one
    if not os.path.exists(RESOURCE_DIR):
        print(f"Error: Resource directory '{RESOURCE_DIR}' not found.")
        return

    pdf_files = [f for f in os.listdir(RESOURCE_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in the '{RESOURCE_DIR}' directory.")
        return

    print(f"Found {len(pdf_files)} PDF(s) to process...")

    for pdf_file in pdf_files:
        print(f"\n--- Processing: {pdf_file} ---")
        pdf_path = os.path.join(RESOURCE_DIR, pdf_file)
        
        # 3. Extract text from the PDF
        print("Extracting text...")
        raw_text = get_pdf_texts([pdf_path])
        if not raw_text:
            print(f"No text extracted from {pdf_file}. Skipping.")
            continue
        print(f"Extracted {len(raw_text)} characters.")

        # 4. Split text into chunks
        print("Splitting text into chunks...")
        text_chunks = get_text_chunks(raw_text)
        print(f"Created {len(text_chunks)} chunks.")

        # 5. Embed and store chunks in ChromaDB
        print("Embedding text and storing in ChromaDB...")
        embed_and_store_chunks(text_chunks, collection)
        print(f"--- Finished processing: {pdf_file} ---")

    print("\nDatabase build process complete!")
    print(f"Total documents in collection: {collection.count()}")

if __name__ == "__main__":
    main()