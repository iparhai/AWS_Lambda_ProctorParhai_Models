# Stage 1: Builder – Build wheels for Python dependencies
FROM python:3.12-slim AS builder

WORKDIR /wheels

# Install build dependencies using apt-get
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake gcc g++ build-essential \
    libcurl4-openssl-dev \
    nodejs npm \
    python3-dev python3-pip python3-venv \
    default-jdk && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and build wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir=/wheels -r requirements.txt

# Stage 2: Application Build – Copy code and prepare runtime files
FROM python:3.12-slim AS app-build

WORKDIR /app

# Copy application code into /app
COPY . .
RUN chmod +x lambda_entry_script.sh

# Stage 3: Minimal Runtime Image
FROM python:3.12-slim

WORKDIR /app

# Install only runtime dependencies using apt-get, plus curl to download RIE
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc g++ \
    nodejs npm \
    python3-dev python3-pip python3-venv \
    default-jdk && \
    rm -rf /var/lib/apt/lists/*

# Download and install AWS Lambda Runtime Interface Emulator (RIE)
RUN curl -Lo /usr/local/bin/aws-lambda-rie \
      https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/latest/download/aws-lambda-rie && \
    chmod +x /usr/local/bin/aws-lambda-rie

# Copy the built wheels from the builder stage
COPY --from=builder /wheels /wheels

# Copy the application code from the app-build stage
COPY --from=app-build /app /app

# Install Python dependencies using the pre-built wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt

# Set the entrypoint for AWS Lambda integration using awslambdaric
ENTRYPOINT [ "/app/lambda_entry_script.sh", "main.handler" ]