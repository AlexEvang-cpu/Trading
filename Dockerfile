# Use Python as the base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Install system dependencies required for TA-Lib
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    wget \
    make \
    libatlas-base-dev \
    ta-lib

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Run the application
CMD ["python", "main.py"]
