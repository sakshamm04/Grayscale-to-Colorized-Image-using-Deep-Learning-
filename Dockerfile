FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    git git-lfs libgl1 libglx-mesa0 libglib2.0-0 \
    libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Fix matplotlib config
ENV MPLCONFIGDIR=/tmp/matplotlib
RUN mkdir -p /tmp/matplotlib && chmod 777 /tmp/matplotlib

RUN git lfs install

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

# CRITICAL: Make sure we run app.py
CMD ["python", "app.py"]
