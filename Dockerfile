# AlphaForge research API image.
# Build:  docker build -t alphaforge .
# Run:    docker run -p 8000:8000 alphaforge
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
COPY apps ./apps
COPY configs ./configs

RUN pip install --upgrade pip \
    && pip install -e ".[api,dashboard,viz]"

EXPOSE 8000 8501

CMD ["alphaforge", "serve-api", "--api-port", "8000"]
