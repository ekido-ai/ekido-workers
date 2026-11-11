# syntax=docker/dockerfile:1
# CPU-only image for the full Python test suite. No GPU needed, and much smaller than the runtime image.
# Run tests with infra/run-tests.sh.
FROM python:3.12-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN pip install --timeout 300 --retries 10 --upgrade "pip>=25.3"
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 300 --retries 10 --resume-retries 20 "torch==2.15.0" --index-url https://download.pytorch.org/whl/cpu
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 300 --retries 10 --resume-retries 20 safetensors numpy eth-account "sqlalchemy>=2" "fastapi>=0.115" httpx pydantic-settings websockets "uvicorn>=0.30" web3
WORKDIR /repo
