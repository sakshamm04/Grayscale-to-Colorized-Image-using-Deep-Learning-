FROM python:3.9-slim

# Install Git and Git LFS for model files
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Configure Git LFS
RUN git lfs install

# Copy requirements and install Python dependencies first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files (including .git if cloning, or LFS files if copying)
COPY . .

# Pull Git LFS files if they exist
RUN if [ -d ".git" ]; then \
        echo "🔄 Pulling Git LFS files..." && \
        git lfs pull; \
    else \
        echo "⚠️  No .git directory found. Make sure LFS files are already present."; \
        ls -la models/ || echo "❌ Models directory not found"; \
    fi

# Verify model files are present and correct size
RUN python -c "\
import os; \
model_path = 'models/colorization_release_v2.caffemodel'; \
if os.path.exists(model_path): \
    size = os.path.getsize(model_path); \
    print(f'✅ Model file found: {size/1024/1024:.1f} MB'); \
    assert size > 100*1024*1024, f'Model file too small: {size} bytes. Run git lfs pull.'; \
else: \
    raise FileNotFoundError('❌ Model file not found. Check Git LFS setup.')\
"

# Set environment variables
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from colorizer import check_model_health; exit(0 if check_model_health() else 1)"

# Start the application
CMD ["python", "app.py"]