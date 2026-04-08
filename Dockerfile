# ─── Supply Disruption Responder ────────────────────────────────────────────
# Build : docker build -t supply-env .
# Run   : docker run supply-env
# With OpenAI agent:
#   docker run -e OPENAI_API_KEY=sk-... supply-env python agent/baseline_agent.py
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Keep Python output unbuffered for cleaner Docker logs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies first (layer-cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY app/       app/
COPY agent/     agent/
COPY scripts/   scripts/
COPY inference.py .

# Default command: run the full evaluation (no API key required)
CMD ["python", "scripts/evaluate.py", "--verbose"]
