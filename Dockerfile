FROM python:3.12-slim AS builder

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache -e "."

FROM python:3.12-slim

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY src/ src/
COPY config/ config/
COPY artifacts/ artifacts/
COPY vendor/ vendor/

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Non-root user
RUN useradd --create-home appuser && \
    mkdir -p /home/appuser/.cache/huggingface && \
    chown -R appuser:appuser /home/appuser/.cache

# Fail the image build if the checked-in gate or vendored runtime is missing.
RUN python -c "import pickle; pickle.load(open('artifacts/kronos/gate_v21.pkl', 'rb')); from vendor.kronos_model import KronosPredictor"
USER appuser

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8003/health')" || exit 1

EXPOSE 8003
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8003"]
