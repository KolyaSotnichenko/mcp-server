# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:${PATH}" \
    UV_LINK_MODE=copy

WORKDIR /app

# System deps: curl for uv install, node & npm for JS execution, bash for bash execution
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates bash nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# Install uv (modern Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy dependency files first (for better Docker layer caching)
COPY pyproject.toml uv.lock ./

# Install project dependencies (respect uv.lock for reproducible builds)
RUN uv sync --frozen --no-dev

# Copy application code
COPY . .

EXPOSE 8000

# Run via uv using the synced virtualenv
CMD ["uv", "run", "mcp-server.py"]