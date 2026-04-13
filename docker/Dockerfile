# Switched to 3.11 for better compatibility with modern Google APIs & Spark
FROM python:3.11-slim

# Install Java 17 AND procps (Spark needs procps for memory management)
RUN apt-get update && apt-get install -y \
    default-jre \
    procps \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements first (make sure 'requests' and 'python-dotenv' are in here!)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set PYTHONPATH so 'import DataCollection' works
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Copy the rest of the project
COPY . .

# Default command
CMD ["python", "src/entry/producer.py"]