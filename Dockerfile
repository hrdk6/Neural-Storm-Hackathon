# syntax=docker/dockerfile:1.7
# Serving image: ONNX Runtime on CPU, no PyTorch. The model bundle is mounted at runtime (never baked in).

FROM python:3.12-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.0 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project --extra dicom
COPY src ./src
COPY web ./web
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-editable --extra dicom

FROM python:3.12-slim-bookworm
RUN groupadd --system app && useradd --system --gid app --home /app app \
    && mkdir -p /models /var/lib/clinictune && chown app:app /var/lib/clinictune
COPY --from=build --chown=app:app /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    CLINICTUNE_BUNDLE_DIR=/models/bundle \
    CLINICTUNE_DATA_DIR=/var/lib/clinictune
USER app
WORKDIR /app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/readyz').status==200 else 1)"
CMD ["clinictune", "serve", "--host", "0.0.0.0", "--port", "8000"]
