import os
import pypdf
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from pathlib import Path # Import the Path object

# --- Explicitly load the .env file ---
# This makes the script work regardless of where you run it from
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file or check the path.")

RESOURCE_DIR = "resource"
DB_PATH = "chroma_db"
COLLECTION_NAME = "medical_textbooks"
MODEL_NAME = "models/embedding-001"

# --- Main Script ---
def build_database():
    print("Starting database build process...")

    if not os.path.exists(RESOURCE_DIR):
        print(f"ERROR: Resource directory '{RESOURCE_DIR}' not found. Please create it and add your PDFs.")
        return

    pdf_files = [f for f in os.listdir(RESOURCE_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in the '{RESOURCE_DIR}' directory.")
        return

    print(f"Found {len(pdf_files)} PDF(s) to process.")

    # 1. Load and Chunk Documents
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)

    for pdf_file in pdf_files:
        pdf_path = os.path.join(RESOURCE_DIR, pdf_file)
        print(f"Processing '{pdf_file}'...")
        try:
            loader = pypdf.PdfReader(pdf_path)
            pdf_text = ""
            for page in loader.pages:
                pdf_text += page.extract_text() or ""
            
            chunks = text_splitter.split_text(pdf_text)
            all_chunks.extend(chunks)
            print(f"-> Extracted {len(chunks)} chunks.")
        except Exception as e:
            print(f"--> ERROR processing '{pdf_file}': {e}")
    
    if not all_chunks:
        print("No text could be extracted from the PDF files. Aborting.")
        return

    print(f"\nTotal chunks to be added to the database: {len(all_chunks)}")

    # 2. Initialize ChromaDB Client and Embedding Function
    print("\nInitializing ChromaDB and embedding function...")
    client = chromadb.PersistentClient(path=DB_PATH)
    gemini_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=GEMINI_API_KEY, model_name=MODEL_NAME
    )

    # 3. Create or Get Collection
    print(f"Getting or creating ChromaDB collection: '{COLLECTION_NAME}'")
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=gemini_ef
    )

    # 4. Add Documents to Collection
    # ChromaDB requires a unique ID for each chunk. We'll generate simple ones.
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    
    print("Adding chunks to the collection... (This may take a while)")
    # ChromaDB can't handle too many documents at once, so we add them in batches.
    batch_size = 100 
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        collection.add(
            documents=batch_chunks,
            ids=batch_ids
        )
        print(f"-> Added batch {i//batch_size + 1}/{(len(all_chunks)-1)//batch_size + 1}")

    print("\nDatabase build process complete!")
    print(f"Total documents in collection: {collection.count()}")

if __name__ == "__main__":
    build_database()