FROM python:3.11-slim

# Install async HTTP libraries
RUN pip install --no-cache-dir \
    bittensor-wallet==4.1.1 \
    httpx[http2] \
    aiohttp

COPY async_host.py /app/async_host.py
COPY track_config.py /app/track_config.py
COPY run.py /app/run.py

WORKDIR /app

CMD ["python", "run.py"]

