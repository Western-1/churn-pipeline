FROM python:3.11-slim

LABEL maintainer="andriy.vlonha.dev@gmail.com"
LABEL description="MLflow Tracking Server"

WORKDIR /mlflow

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install MLflow and dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    mlflow==2.8.1 \
    psycopg2-binary==2.9.9 \
    boto3==1.29.7 \
    cryptography==41.0.7

# Create non-root user
RUN useradd -m -u 1000 mlflow && \
    chown -R mlflow:mlflow /mlflow

USER mlflow

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run MLflow server
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "${MLFLOW_BACKEND_STORE_URI}", \
     "--default-artifact-root", "${MLFLOW_ARTIFACT_ROOT}"]
