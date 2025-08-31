#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "INFO: Build script started."

# Install Python dependencies
echo "INFO: Installing requirements from requirements.txt..."
pip install -r requirements.txt
echo "INFO: Requirements installed successfully."

# Download the database
echo "INFO: Running database download script..."
python download_db.py
echo "INFO: Database download script finished."

echo "INFO: Build script completed successfully."