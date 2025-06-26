FROM python:3.11-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
COPY . .

RUN uv sync

ENV PYTHONPATH=/app/src

CMD ["uv", "run", "python", "main.py"]