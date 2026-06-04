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
        gnupg \
        libgomp1 \
        ripgrep \
        unzip \
        zip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-mcp.txt /tmp/

RUN if [ -n "$PIP_INDEX_URL" ]; then \
        python -m pip config set global.index-url "$PIP_INDEX_URL"; \
    fi \
    && python -m pip install --no-compile \
        -r /tmp/requirements.txt \
        -r /tmp/requirements-mcp.txt \
    && rm /tmp/requirements.txt /tmp/requirements-mcp.txt \
    && find /usr/local/lib/python3.11/site-packages -type d -name "__pycache__" -prune -exec rm -rf '{}' + \
    && find /usr/local/lib/python3.11/site-packages -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete \
    && find /usr/local/lib/python3.11/site-packages -type d \( -name "test" -o -name "tests" \) -prune -exec rm -rf '{}' +

RUN set -eux; \
    mkdir -p /etc/apt/keyrings; \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
        | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg; \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_22.x nodistro main" \
        > /etc/apt/sources.list.d/nodesource.list; \
    apt-get update -o Acquire::Retries=3 -o Acquire::http::Timeout=30 \
    && apt-get install -y --no-install-recommends \
        fonts-dejavu-core \
        fonts-noto-cjk \
        fonts-wqy-microhei \
        fonts-wqy-zenhei \
        ghostscript \
        graphviz \
        libcairo2 \
        libgdk-pixbuf-2.0-0 \
        libharfbuzz0b \
        libpango-1.0-0 \
        libpangocairo-1.0-0 \
        libpangoft2-1.0-0 \
        libreoffice \
        nodejs \
        pandoc \
        poppler-utils \
        qpdf \
        shared-mime-info \
        tesseract-ocr \
        tesseract-ocr-eng \
        wget \
        wkhtmltopdf \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g --omit=dev --no-audit --no-fund \
        @modelcontextprotocol/server-filesystem@2025.12.18 \
        @sylphx/pdf-reader-mcp@2.3.0 \
        docx \
        pdf-lib \
        pdfjs-dist@3.11.174 \
        pptxgenjs \
    && npm cache clean --force

ENV NODE_PATH=/usr/local/lib/node_modules:/usr/lib/node_modules

COPY requirements-pdf-extra.txt /tmp/gdpval-pdf-extra-requirements.txt

RUN python -m pip install --no-compile -r /tmp/gdpval-pdf-extra-requirements.txt \
    && rm /tmp/gdpval-pdf-extra-requirements.txt \
    && find /usr/local/lib/python3.11/site-packages -type d -name "__pycache__" -prune -exec rm -rf '{}' + \
    && find /usr/local/lib/python3.11/site-packages -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete \
    && find /usr/local/lib/python3.11/site-packages -type d \( -name "test" -o -name "tests" \) -prune -exec rm -rf '{}' +

WORKDIR /app/workspace/gdpevals

CMD ["python", "--version"]
