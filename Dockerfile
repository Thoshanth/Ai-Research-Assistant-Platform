# Use official Python 3.11 slim image
# slim = smaller size, no unnecessary OS packages
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (Docker caches this layer)
# If requirements don't change, this layer is reused
# making rebuilds much faster
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads logs chroma_db

# Expose port 8000
EXPOSE 8000

# Start command
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]