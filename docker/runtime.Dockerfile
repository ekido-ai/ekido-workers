# syntax=docker/dockerfile:1
# Ekido training runtime. One shared image, because disk is limited on the demo laptop.
FROM python:3.12-slim

ARG TORCH_VERSION=2.15.0

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/hf_cache

# Downloads on a flaky connection gave SHA-256 mismatches on both PyPI and pypi.nvidia.com, alongside
# SSLEOFError in the log. Likely cause: pip 25.0.1 not noticing a connection cut mid-file.
# So use a newer pip that can resume. Hash checks stay on: never disable them to get past a mismatch.
RUN pip install --timeout 300 --retries 10 --upgrade "pip>=25.3"

# Installed from PyPI, not download.pytorch.org, whose index points at the much slower pypi.nvidia.com.
# The cache mount keeps finished downloads across failed builds and adds nothing to the image.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 300 --retries 10 --resume-retries 20 "torch==${TORCH_VERSION}"
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 300 --retries 10 --resume-retries 20 "transformers==5.23.0" "peft==0.22.0" "safetensors" "datasets" "numpy" "eth-account"

RUN useradd --create-home --uid 1000 gpuchain && mkdir -p /hf_cache /work && chown -R gpuchain /hf_cache /work
USER gpuchain
WORKDIR /work
