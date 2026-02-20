FROM python:3.13-slim

WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir .
COPY . .

ENV PYTHONPATH=/app/src
CMD ["python", "-m", "cli.main", "run"]
