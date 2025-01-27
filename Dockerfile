# Use the official Python slim image as a base
FROM python:3.11-slim

# Set environment variables to prevent .pyc files and ensure UTF-8 support
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV TOKENIZERS_PARALLELISM=false
ENV OPENAI_API_KEY="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"


# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc build-essential curl && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -i https://pypi.org/simple --default-timeout=100 -r requirements.txt && \
    apt-get remove -y gcc build-essential && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the entire app into the container
COPY . /app

# Expose the application port
EXPOSE 8000

# Command to run the application
CMD ["streamlit", "run", "tariff_agent.py", "--server.address=0.0.0.0", "--server.port=8000"]