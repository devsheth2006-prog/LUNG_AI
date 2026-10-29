# CPU image running the Streamlit UI (default) or the FastAPI service.
#   docker build -t ct-cancer-ai .
#   docker run -p 8501:8501 -v $PWD/models:/app/models ct-cancer-ai                 # UI
#   docker run -p 8000:8000 -v $PWD/models:/app/models ct-cancer-ai api             # API
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src TORCH_HOME=/app/.torch
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt
COPY . .
RUN useradd -m appuser && mkdir -p models outputs mlruns .torch && chown -R appuser /app
USER appuser
EXPOSE 8501 8000
HEALTHCHECK CMD curl -fs http://localhost:8501/_stcore/health || curl -fs http://localhost:8000/health || exit 1
COPY --chown=appuser docker-entrypoint.sh /usr/local/bin/entry
ENTRYPOINT ["sh", "/usr/local/bin/entry"]
CMD ["ui"]
