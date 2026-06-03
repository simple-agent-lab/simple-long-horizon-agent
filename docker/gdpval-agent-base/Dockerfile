# syntax=docker/dockerfile:1

FROM python:3.11-slim-bookworm

ARG DEBIAN_MIRROR=""
ARG PIP_INDEX_URL=""

LABEL org.opencontainers.image.title="GDPval-Agent-Base"
LABEL org.opencontainers.image.description="Python 3.11 base image with common GDPVal task libraries for Simple Agent Lab local-docker runs."

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_NO_CACHE_DIR=1
ENV PIP_RETRIES=5

RUN if [ -n "$DEBIAN_MIRROR" ]; then \
        find /etc/apt -type f \( -name "*.list" -o -name "*.sources" \) -print0 \
        | xargs -0 sed -i \
            -e "s|http://deb.debian.org|$DEBIAN_MIRROR|g" \
            -e "s|http://security.debian.org|$DEBIAN_MIRROR|g"; \
    fi \
    && apt-get update -o Acquire::Retries=3 -o Acquire::http::Timeout=30 \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        file \
        git \
        libgomp1 \
        ripgrep \
        unzip \
        zip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/gdpval-requirements.txt

RUN if [ -n "$PIP_INDEX_URL" ]; then \
        python -m pip config set global.index-url "$PIP_INDEX_URL"; \
    fi \
    && python -m pip install -r /tmp/gdpval-requirements.txt \
    && rm /tmp/gdpval-requirements.txt

WORKDIR /app/workspace/gdpevals

CMD ["python", "--version"]
