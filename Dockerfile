# Python runtime
FROM python:3.10-slim

# Application directory
WORKDIR /app

# Install dependencies first for better build caching
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Run the research agent
ENTRYPOINT ["python", "main.py"]
