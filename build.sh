#!/usr/bin/env bash
# exit on error
set -o errexit

echo "--- Starting comprehensive build script ---"

# Update package lists
echo "Updating package lists..."
apt-get update -y

# Install a comprehensive set of system dependencies for OpenCV and multimedia
echo "Installing all potential system dependencies..."
apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg

# Install Python dependencies from requirements.txt
echo "Installing Python packages..."
pip install -r requirements.txt

echo "--- Build script finished successfully ---"
