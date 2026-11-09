FROM python:3.11-slim

ARG WITH_WHISPER=0
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    VAULT_PATH=/vault DATA_DIR=/data HOME=/data

WORKDIR /app
COPY requirements.txt requirements-optional.txt ./
RUN pip install -r requirements.txt && \
    if [ "$WITH_WHISPER" = "1" ]; then pip install "faster-whisper>=1.0"; fi
COPY app ./app
COPY prompts ./prompts

EXPOSE 8765
CMD ["uvicorn", "app.main:get_app", "--factory", "--host", "0.0.0.0", "--port", "8765"]
