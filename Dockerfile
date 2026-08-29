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

# Build the venv and install dependencies first, using only the dependency
# manifest. This layer is cached across rebuilds unless the dependency list
# changes — so a source-only change (the common case) re-runs only the final
# `pip install .` of the package itself, not the ~90 transitive deps.
COPY pyproject.toml README.md ./

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
# Isolate requirements: extract the dependency set from pyproject into a
# throwaway requirements file, so this layer keys off the
# [project.dependencies] block alone and not on the package source.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip wheel \
 && python - <<'PY' > /tmp/reqs.txt
import tomllib
deps = tomllib.load(open("pyproject.toml","rb"))["project"]["dependencies"]
for d in deps:
    print(d)
PY
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/reqs.txt

# Copy the package source ONLY after the deps layer is cached, so source changes
# do not invalidate the dependency install above.
COPY mm_gateway ./mm_gateway
# Install the gateway package itself (uses the deps above; only re-runs when the
# mm_gateway source tree changes).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps .

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

# Note: the image installs only the base dependencies (the
# [project.dependencies] block) — it does NOT include the optional [socks] extra
# (socksio / python-socks / aiohttp-socks). A deployment that routes outbound
# traffic through a SOCKS5 proxy — or that sets an explicit outbound_proxy on a
# dashscope backend (the aiohttp path routes any explicit proxy, HTTP or SOCKS,
# through aiohttp-socks) — must add it, e.g. build a child image that runs
# `pip install mm-gateway[socks]`. HTTP (CONNECT) proxies on the httpx/WS paths
# need no extra dep.
CMD ["mm-gateway"]
