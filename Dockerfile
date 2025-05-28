# Use official Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install OpenCV dependencies (fixes libGL.so.1 error)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY webapp /app/webapp
COPY model /app/model
COPY static /app/static
COPY templates /app/templates
COPY .env /app/.env

# Expose the port (default FastAPI port)
EXPOSE 8000

# Start the FastAPI app with Uvicorn
CMD ["uvicorn", "webapp.model:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
