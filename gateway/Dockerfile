FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY gateway/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
COPY gateway ./gateway
# Model metadata (configs + ONNX) so the gateway knows models/versions and can run them in-process when TRITON_URL is unset.
COPY model_repository ./model_repository
USER app
EXPOSE 8080
CMD ["sh", "-c", "uvicorn gateway.app.main:app --host 0.0.0.0 --port 8080 --workers ${WORKERS:-2} --no-access-log"]
