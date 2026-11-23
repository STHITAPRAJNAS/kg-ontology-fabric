# Production multi-stage Dockerfile for self-hosted Docker Compose or Amazon EKS
FROM python:3.11-slim AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

# Install service with Presidio PII, AWS Neptune/Bedrock, MCP, Graph Analytics, Auth & OpenTelemetry extras
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[pii,aws,mcp,graph-analytics,otel,auth]" && \
    python -m spacy download en_core_web_lg

FROM python:3.11-slim AS runtime

WORKDIR /app
RUN groupadd -r kgfabric && useradd -r -g kgfabric kgfabric

# Copy installed packages + pre-baked spaCy/Presidio models so private EKS pods need zero egress
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY src ./src
COPY pyproject.toml ./

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PII_REDACTION_MODE=replace_tag \
    KG_FABRIC_DOMAIN_PACK=cloud_sre \
    KG_FABRIC_AUTH_MODE=api_key \
    UVICORN_WORKERS=2

USER kgfabric
EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health')"

CMD ["kg-fabric", "serve", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
