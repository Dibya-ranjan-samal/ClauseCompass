FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir uv && uv sync --frozen --no-dev

# Bake the models into the image. Without this, cold start hits the network and
# a Hugging Face rate limit becomes a failed deploy - which cost this project
# most of a day on Day 2.
RUN uv run python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('BAAI/bge-reranker-base')"

COPY evals/ ./evals/
COPY dashboard.py ablation.py run_audit.py ./

ENV PYTHONPATH=/app/src
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
  CMD curl -f http://localhost:8000/healthz || exit 1
CMD ["uv", "run", "uvicorn", "clausecompass.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]
