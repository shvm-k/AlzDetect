# AlzDetect — container image (works on Hugging Face Spaces, Render, Railway, Fly).
# Hugging Face Spaces expects the app to listen on port 7860.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HF_HOME=/tmp/hf

WORKDIR /app

# System deps for Pillow / TensorFlow runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Writable model cache for MODEL_URL downloads.
RUN mkdir -p backend/models && chmod -R 777 backend/models

WORKDIR /app/backend
EXPOSE 7860

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}
