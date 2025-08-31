import os
import time
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import ffmpeg
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configuration & Initialization ---
# Use a more robust way to find and load the .env file
script_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(script_dir, '.env')
load_dotenv(dotenv_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it as an environment variable.")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# --- Flask App Initialization ---
app = Flask(__name__, template_folder='.')

# --- THE FIX IS HERE ---
# Use the /tmp directory for uploads, which is writable in container environments
UPLOAD_FOLDER = '/tmp/uploads'
# --- END OF FIX ---

DB_PATH = os.path.join(script_dir, "render_disk", "chroma_db")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DB_PATH'] = DB_PATH

# Create the uploads folder at startup
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- ChromaDB Connection ---
collection = None
try:
    client = chromadb.PersistentClient(path=app.config['DB_PATH'])
    collection = client.get_collection(name="medical_manuals")
    print("Successfully connected to ChromaDB collection.")
except Exception as e:
    print(f"Error connecting to ChromaDB: {e}. The RAG functionality will be disabled.")
    collection = None


# --- Core RAG & AI Functions ---
def extract_audio_from_video(video_path):
    """Extracts audio from a video file and saves it as an MP3."""
    try:
        audio_path = os.path.join(os.path.dirname(video_path), "extracted_audio.mp3")
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, acodec='libmp3lame', audio_bitrate='192k')
            .overwrite_output()
            .run(quiet=True)
        )
        print(f"Audio extracted successfully to {audio_path}")
        return audio_path
    except ffmpeg.Error as e:
        print(f"Error extracting audio with ffmpeg: {e.stderr.decode()}")
        return None

def transcribe_audio(audio_path):
    """Transcribes audio using the Gemini API."""
    print("Uploading audio file for transcription...")
    try:
        audio_file = genai.upload_file(path=audio_path)
        print("Audio file uploaded. Requesting transcription...")
        
        prompt = "Transcribe the following audio recording of a surgical procedure. Provide only the text of the transcription."
        response = model.generate_content([prompt, audio_file])
        
        genai.delete_file(audio_file.name)
        print("Transcription complete. Temporary audio file deleted from cloud.")
        
        return response.text
    except Exception as e:
        print(f"Error during transcription: {e}")
        return f"Error: Could not transcribe audio. Details: {e}"

def get_rag_analysis(transcript, system_prompt, user_prompt):
    """Performs the RAG query and generates the final analysis."""
    if not collection:
        return "Error: ChromaDB connection failed. Cannot perform RAG analysis."
    
    print("Performing RAG retrieval...")
    try:
        # 1. Retrieval: Query ChromaDB for relevant context
        results = collection.query(
            query_texts=[transcript],
            n_results=5  # Retrieve the top 5 most relevant chunks
        )
        
        retrieved_docs = results.get('documents', [[]])[0]
        context = "\n\n".join(retrieved_docs)
        print(f"Retrieved {len(retrieved_docs)} context chunks.")

        # 2. Augmentation & 3. Generation
        # Manually combine prompts for compatibility with older library versions
        combined_prompt = f"""{system_prompt}

        **Retrieved Medical Context:**
        ---
        {context}
        ---

        {user_prompt}
        """

        print("Sending augmented prompt to Gemini for final analysis...")
        response = model.generate_content(combined_prompt)
        
        return response.text
    except Exception as e:
        print(f"An error occurred during the RAG workflow: {e}")
        return f"Error during RAG analysis: {e}"

# --- API Endpoints ---
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
    system_prompt = request.form.get('system_prompt', 'No system prompt provided.')
    user_prompt = request.form.get('user_prompt', 'No user prompt provided.')

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    video_path = None
    audio_path = None
    try:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(video_path)
        print(f"Video saved to {video_path}")

        audio_path = extract_audio_from_video(video_path)
        if not audio_path:
            return jsonify({'error': 'Failed to extract audio from video'}), 500
        
        transcript = transcribe_audio(audio_path)
        analysis = get_rag_analysis(transcript, system_prompt, user_prompt)
        
        return jsonify({'analysis': analysis, 'transcript': transcript})

    except Exception as e:
        print(f"An unexpected error occurred in /analyze: {e}")
        return jsonify({'error': f'An unexpected server error occurred: {e}'}), 500

    finally:
        # Clean up temporary files
        try:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
            print("Cleaned up temporary files.")
        except OSError as e:
            print(f"Error cleaning up files: {e}")

# --- Main Execution ---
if __name__ == '__main__':
    # This block is for local development only
    # Gunicorn will be the server in production
    app.run(debug=True, port=5000)

