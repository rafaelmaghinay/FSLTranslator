FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PORT=8000

# Install system dependencies
RUN apt-get clean && \
    apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libfontconfig1 \
    libice6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads/images uploads/sequences uploads/videos uploads/temp models

# ⬅️ IMPORTANT: Cloud Run sets PORT env variable
EXPOSE ${PORT}

# ⬅️ Production command (no --reload)
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1