FROM python:3.9-slim

# Install Java 17 (Required for Spark 3.5+)
RUN apt-get update && apt-get install -y default-jre && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set PYTHONPATH so 'import DataCollection' works from the entry scripts
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Copy the rest of the project
COPY . .

# Default command (can be overridden in docker-compose)
CMD ["python", "src/entry/producer.py"]