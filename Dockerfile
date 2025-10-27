# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=on \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

# Install build tooling
RUN pip install --no-cache-dir --upgrade pip build

# Copy only files needed to build the wheel first to leverage Docker layer caching
COPY pyproject.toml README.md ./
COPY vct ./vct

# Build the project wheel
RUN python -m build --wheel --outdir dist


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_NO_CACHE_DIR=on

# Create an unprivileged user to run the application
RUN useradd --system --create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

# Install the application wheel and runtime dependencies only
COPY --from=builder /app/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm -f /tmp/*.whl

EXPOSE 8000
ENV VCT_SIMULATE=1

USER appuser

CMD ["uvicorn", "vct.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
