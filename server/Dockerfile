# Supply Disruption Responder — HF Spaces Dockerfile
# HF Spaces requires port 7860

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install server dependencies
COPY server/requirements.txt /tmp/server_requirements.txt
RUN pip install --no-cache-dir -r /tmp/server_requirements.txt

# Copy the full project
COPY . /app/

# HF Spaces uses port 7860
EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
