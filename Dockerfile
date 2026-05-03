FROM python:3.11-slim

WORKDIR /app

# Fix Debian Trixie : libgl1-mesa-glx remplacé par libgl1
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code source
COPY . .

# Pre-télécharge le modèle HuggingFace au build
RUN python -c "from transformers import pipeline; pipeline('image-classification', model='linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification', top_k=3)"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
