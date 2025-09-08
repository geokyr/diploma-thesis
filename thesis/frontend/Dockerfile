FROM python:3.12.11-slim

RUN pip install --no-cache-dir uv \
    && apt-get update \
    && apt-get install -y curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra frontend --no-install-project

COPY thesis ./thesis
RUN uv pip install -e .

EXPOSE 8080

CMD ["uv", "run", "python", "thesis/frontend/main.py"]
