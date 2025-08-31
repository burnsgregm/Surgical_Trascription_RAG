import os
import requests
import zipfile
import io

# --- Configuration ---
# IMPORTANT: Replace this with the direct download link to your zipped chroma_db folder.
# This link must be a direct download link, not a preview page.
# See README.md for instructions on how to get this link from Google Drive.
# https://drive.google.com/file/d/1erTv4Xu1v76OTgIlldCAUdz7VfbZRebo/view?usp=sharing
# https://drive.google.com/uc?export=download&id=1erTv4Xu1v76OTgIlldCAUdz7VfbZRebo
DB_DOWNLOAD_URL = "https://drive.google.com/uc?export=download&id=1erTv4Xu1v76OTgIlldCAUdz7VfbZRebo"

# The path where Render's persistent disk is mounted.
# The database will be downloaded and unzipped here.
RENDER_DISK_PATH = "/var/data"
DB_TARGET_PATH = os.path.join(RENDER_DISK_PATH, "chroma_db")

def download_and_unzip_db():
    """
    Downloads the pre-built ChromaDB from a cloud URL and unzips it
    into the specified target directory on Render's persistent disk.
    """
    print("--- Starting Database Download and Unzip Process ---")

    if not DB_DOWNLOAD_URL or "YOUR_DIRECT_DOWNLOAD_LINK_HERE" in DB_DOWNLOAD_URL:
        print("ERROR: DB_DOWNLOAD_URL is not set in download_db.py. Skipping download.")
        # In a real deployment, you might want this to be a hard failure.
        # For flexibility, we'll allow the app to start, but RAG will fail.
        return

    print(f"Target directory: {DB_TARGET_PATH}")
    if not os.path.exists(RENDER_DISK_PATH):
        print(f"Creating Render disk mount directory: {RENDER_DISK_PATH}")
        os.makedirs(RENDER_DISK_PATH)

    try:
        print(f"Downloading database from {DB_DOWNLOAD_URL}...")
        response = requests.get(DB_DOWNLOAD_URL, stream=True)
        response.raise_for_status()  # This will raise an exception for bad status codes

        print("Download complete. Unzipping database...")
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(RENDER_DISK_PATH)
        
        # After unzipping, the contents will be in a folder like 'render_disk/chroma_db'.
        # We need to rename the unzipped folder to just 'chroma_db'.
        unzipped_folder_name = z.namelist()[0].split('/')[0] # Get the top-level folder name from the zip
        unzipped_path = os.path.join(RENDER_DISK_PATH, unzipped_folder_name)
        
        if os.path.exists(unzipped_path) and unzipped_path != DB_TARGET_PATH:
            print(f"Renaming unzipped folder from '{unzipped_path}' to '{DB_TARGET_PATH}'")
            os.rename(unzipped_path, DB_TARGET_PATH)
        
        print("Database successfully unzipped and placed in target directory.")

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to download the database. Error: {e}")
    except zipfile.BadZipFile:
        print("ERROR: The downloaded file is not a valid zip file. Check your download link.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    download_and_unzip_db()
    print("--- Database Download Script Finished ---")

