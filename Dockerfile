FROM python:3.9-slim

# Install system dependencies for scikit-learn
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "main:app"]
