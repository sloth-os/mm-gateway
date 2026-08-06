# syntax=docker/dockerfile:1
#
# Multi-stage build for mm-gateway.
#
# The builder stage compiles any C extensions (a couple of the provider SDK
# transitive deps ship sdist-only for some arches) into an isolated venv; the
# runtime stage copies just that venv onto a slim image. No source, tests, or
# local config are copied into the runtime image — config is supplied at run
# time via env vars (the legacy env-var layout) or a mounted mm-gateway.yaml.

ARG PYTHON_VERSION=3.12

# ---- builder ----------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only what `pip install .` needs. README is referenced by pyproject
# (readme = "README.md"); the package source is discovered via the
# [tool.setuptools.packages.find] include rule.
COPY pyproject.toml README.md ./
COPY mm_gateway ./mm_gateway

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN pip install --upgrade pip wheel && pip install .

# ---- runtime ----------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8000

# Non-root user for the running process.
RUN groupadd -r app && useradd -r -g app -d /app app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
USER app

EXPOSE 8000

# Lightweight liveness probe using the stdlib (no curl/wget on slim). The
# gateway serves GET /health unconditionally.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)" || exit 1

CMD ["mm-gateway"]
