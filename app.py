import os
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import chromadb
# The specific embedding_functions import is no longer needed here
import subprocess
import uuid

# --- CONFIGURATION ---
# More robustly load environment variables
script_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(script_dir, '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it as an environment variable.")

# Configure the Gemini client
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# --- DATABASE AND FILE PATHS ---
DB_DIRECTORY = os.path.join(script_dir, "render_disk/chroma_db")
COLLECTION_NAME = "medical_manuals"
UPLOAD_FOLDER = os.path.join(script_dir, 'uploads')

# --- FLASK APP AND DB INITIALIZATION ---
app = Flask(__name__, template_folder='.')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    try:
        os.makedirs(UPLOAD_FOLDER)
    except PermissionError:
        print("CRITICAL ERROR: Could not create the 'uploads' directory. Check folder permissions.")
    except Exception as e:
        print(f"CRITICAL ERROR: An unexpected error occurred creating 'uploads' directory: {e}")

# Global variable to hold the ChromaDB collection
collection = None

def initialize_database():
    """Initializes and returns the ChromaDB collection."""
    global collection
    try:
        if not os.path.exists(DB_DIRECTORY):
            print(f"Error: ChromaDB directory not found at {DB_DIRECTORY}")
            return None
            
        client = chromadb.PersistentClient(path=DB_DIRECTORY)
        
        # --- FIX IS HERE: We no longer specify an embedding function ---
        # This tells ChromaDB to use its default model (all-MiniLM-L6-v2)
        # which now matches the build script.
        collection = client.get_collection(
            name=COLLECTION_NAME
        )
        
        print("Successfully connected to ChromaDB collection.")
        return collection
    except Exception as e:
        # Provide a more specific error for the collection not existing
        if "does not exist" in str(e):
             print(f"Error connecting to ChromaDB: Collection [{COLLECTION_NAME}] does not exist.")
        else:
            print(f"An unexpected error occurred during ChromaDB initialization: {e}")
        return None

# Initialize the database when the application starts
collection = initialize_database()


# --- CORE FUNCTIONS ---

def extract_audio_from_video(video_path):
    """Extracts audio using ffmpeg and returns the path to the audio file."""
    try:
        audio_filename = f"{uuid.uuid4()}.mp3"
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
        
        command = [
            'ffmpeg',
            '-i', video_path,
            '-q:a', '0',
            '-map', 'a',
            audio_path
        ]
        
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Audio extracted successfully to {audio_path}")
        return audio_path
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg Error: {e.stderr}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during audio extraction: {e}")
        return None


def transcribe_audio(audio_path):
    """Transcribes the given audio file using the Gemini API."""
    print(f"Uploading for transcription: {audio_path}")
    try:
        audio_file = genai.upload_file(path=audio_path)
        prompt = "Transcribe the following audio recording from a surgical procedure."
        response = model.generate_content([prompt, audio_file], request_options={'timeout': 600})
        
        # Clean up the uploaded file from the server
        genai.delete_file(audio_file.name)
        
        return response.text
    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        return f"Error during transcription: {e}"


def get_rag_analysis(transcript, system_prompt, user_prompt):
    """Performs RAG analysis by retrieving context from ChromaDB and querying Gemini."""
    if not collection:
        print("ChromaDB connection failed. Cannot perform RAG analysis.")
        return "Error: ChromaDB connection failed. Cannot perform RAG analysis."

    try:
        # Step 1: Retrieve relevant documents from ChromaDB
        # The query will now automatically use the correct default model
        results = collection.query(
            query_texts=[transcript],
            n_results=5
        )
        retrieved_docs = "\n\n---\n\n".join(results['documents'][0])
        
        # Step 2: Augment the prompt with the retrieved context
        combined_prompt = f"""
{system_prompt}

**Medical Context from Documentation:**
---
{retrieved_docs}
---

**User's Analysis Query:**
{user_prompt}

**Surgical Transcript to Analyze:**
---
{transcript}
---

Based *only* on the provided Medical Context, perform the analysis requested in the User's Query on the Surgical Transcript.
"""
        
        # Step 3: Generate the final analysis
        response = model.generate_content(combined_prompt, request_options={'timeout': 600})
        return response.text
        
    except Exception as e:
        print(f"An error occurred during the RAG workflow: {e}")
        return f"An error occurred during the RAG workflow: {e}"


# --- API ENDPOINTS ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze_video():
    """API endpoint to receive a video, process it, and return an analysis."""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file part'}), 400

    file = request.files['video']
    system_prompt = request.form.get('system_prompt', 'No system prompt provided')
    user_prompt = request.form.get('user_prompt', 'No user prompt provided')

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    video_path = None
    audio_path = None
    try:
        video_filename = f"{uuid.uuid4()}-{file.filename}"
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        file.save(video_path)

        audio_path = extract_audio_from_video(video_path)
        if not audio_path:
            return jsonify({'error': 'Failed to extract audio from video'}), 500
        
        transcript = transcribe_audio(audio_path)
        if "Error during transcription" in transcript:
             return jsonify({'error': transcript}), 500

        analysis = get_rag_analysis(transcript, system_prompt, user_prompt)
        
        return jsonify({
            'transcript': transcript,
            'analysis': analysis
        })

    except Exception as e:
        print(f"An unexpected server error occurred: {e}")
        return jsonify({'error': f'An unexpected server error occurred: {e}'}), 500

    finally:
        # Clean up temporary files
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
        print("Cleaned up temporary files.")


# --- MAIN EXECUTION ---
if __name__ == '__main__':
    # Use Gunicorn for production, but this is for local dev
    app.run(debug=True, port=5000)

