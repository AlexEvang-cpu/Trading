# Use Python as the base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir numpy==1.23.5
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Run the application
CMD ["python", "main.py"]
