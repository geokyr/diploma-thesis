FROM python:3.12.11-slim

RUN pip install --no-cache-dir uv \
    && apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra predictor --no-install-project

COPY thesis ./thesis
RUN uv pip install -e .

EXPOSE 8001 8002 8003
CMD ["uv", "run", "python", "-m", "thesis.predictor.main"]
