FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

RUN pip cache purge && \
    rm -rf /root/.cache/pip

WORKDIR /sandbox

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
