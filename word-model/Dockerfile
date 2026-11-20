FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/word-model

COPY word-model/requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt

COPY word-model/wordmodel ./wordmodel
COPY word-model/server ./server
COPY word-model/models ./models

RUN groupadd --gid 10003 word-ai \
    && useradd --uid 10003 --gid word-ai --create-home word-ai \
    && chown -R word-ai:word-ai /app/word-model

USER word-ai:word-ai

ENV HANDPRACTICE_WORD_HOST=0.0.0.0 \
    HANDPRACTICE_WORD_PORT=8767 \
    HANDPRACTICE_WORD_PATH=/word \
    HANDPRACTICE_WORD_CHECKPOINT=/app/models/word-19-v1/best_model.pt \
    HANDPRACTICE_WORD_TORCH_THREADS=2 \
    HANDPRACTICE_WORD_INFERENCE_WORKERS=1

EXPOSE 8767

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=5 \
    CMD ["python", "-c", "from websockets.sync.client import connect; ws=connect('ws://127.0.0.1:8767/word', open_timeout=3); ws.close()"]

CMD ["python", "-m", "server.main"]
