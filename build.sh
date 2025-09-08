#!/usr/bin/env bash
# exit on error
set -o errexit

# Install all necessary system dependencies for the full OpenCV library
echo "Installing system dependencies..."
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev

# Install Python dependencies from requirements.txt
echo "Installing Python packages..."
pip install -r requirements.txt

echo "Build script finished successfully."