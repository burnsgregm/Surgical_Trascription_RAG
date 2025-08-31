import os
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from pathlib import Path
import ffmpeg
import chromadb
from chromadb.utils import embedding_functions

# --- Explicitly load the .env file ---
# This robustly finds the .env file in the same directory as the script.
script_dir = Path(__file__).parent.resolve()
env_path = script_dir / '.env'
load_dotenv(dotenv_path=env_path)

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it as an environment variable.")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# --- ChromaDB Configuration ---
# Use Render's standard persistent disk mount path.
# When running locally, this will look in a 'render_disk' folder.
DB_PATH = os.environ.get("RENDER_DISK_PATH", "render_disk/chroma_db")
COLLECTION_NAME = "medical_textbooks"
MODEL_NAME = "models/embedding-001"

# --- Flask App Initialization ---
app = Flask(__name__, template_folder='.')
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- ChromaDB Client Initialization ---
print("Initializing ChromaDB client...")
collection = None
if os.path.exists(DB_PATH):
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        gemini_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=GEMINI_API_KEY, model_name=MODEL_NAME
        )
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=gemini_ef)
        print("ChromaDB collection loaded successfully.")
    except Exception as e:
        print(f"Error loading ChromaDB collection: {e}")
else:
    print(f"FATAL: Database path not found at '{DB_PATH}'. The database must be built first.")


# --- Core Functions ---

def extract_audio_from_video(video_path):
    print(f"Extracting audio from {video_path}...")
    try:
        audio_path = video_path + ".mp3"
        (
            ffmpeg.input(video_path)
            .output(audio_path, acodec='libmp3lame', audio_bitrate='192k')
            .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
        )
        return audio_path
    except ffmpeg.Error as e:
        print("FFmpeg stderr:", e.stderr.decode('utf8'))
        return None

def transcribe_audio(audio_path):
    print(f"Uploading {audio_path} for transcription...")
    audio_file = genai.upload_file(path=audio_path)
    
    prompt = "Please transcribe the following audio file from a surgical procedure. Output only the raw text of the transcription."
    
    print("Sending request to Gemini for transcription...")
    response = model.generate_content([prompt, audio_file], request_options={'timeout': 600})
    
    genai.delete_file(audio_file.name)
    print("Transcription complete and uploaded file deleted.")
    return response.text

def get_rag_analysis(transcript, system_prompt, user_prompt):
    if not collection:
        raise ConnectionError("ChromaDB connection failed. Check server logs.")
        
    print("Querying ChromaDB for relevant medical context...")
    results = collection.query(
        query_texts=[transcript],
        n_results=5
    )
    retrieved_chunks = results['documents'][0]
    context = "\n\n---\n\n".join(retrieved_chunks)
    print(f"Retrieved {len(retrieved_chunks)} chunks from the database.")

    # <<< START OF FIX FOR OLDER LIBRARY VERSION >>>
    # Manually combine the system prompt with the rest of the prompt content.
    final_prompt = f"""
{system_prompt}

**User's Query:**
{user_prompt}

**Medical Context from Textbooks:**
---
{context}
---

**Surgical Transcript to Analyze:**
---
{transcript}
---
"""
    # <<< END OF FIX >>>

    print("Sending augmented prompt to Gemini for final analysis...")
    # Call generate_content without the 'system_instruction' parameter.
    response = model.generate_content(final_prompt)
    return response.text


# --- API Endpoints ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file part'}), 400

    file = request.files['video']
    system_prompt = request.form.get('system_prompt', 'No system prompt provided.')
    user_prompt = request.form.get('user_prompt', 'No user prompt provided.')

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(video_path)
        
        audio_path = extract_audio_from_video(video_path)
        if not audio_path:
            return jsonify({'error': 'Failed to extract audio. Check server logs.'}), 500
        
        try:
            transcript = transcribe_audio(audio_path)
            analysis = get_rag_analysis(transcript, system_prompt, user_prompt)
            
            return jsonify({
                'analysis': analysis,
                'transcript': transcript
            })
        except Exception as e:
            print(f"An error occurred during the RAG workflow: {e}")
            return jsonify({'error': f'An error occurred: {e}'}), 500
        finally:
            if os.path.exists(video_path): os.remove(video_path)
            if os.path.exists(audio_path): os.remove(audio_path)
            print("Cleaned up temporary local files.")

    return jsonify({'error': 'An unknown error occurred'}), 500

# --- Main Execution ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)
