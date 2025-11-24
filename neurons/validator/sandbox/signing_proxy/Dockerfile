FROM python:3.11-slim

# Install async HTTP libraries
RUN pip install --no-cache-dir \
    bittensor-wallet \
    httpx[http2] \
    aiohttp

# Copy both sync and async versions (async is now default)
COPY host.py /app/host.py
COPY async_host.py /app/async_host.py
COPY run.py /app/run.py

WORKDIR /app

CMD ["python", "run.py"]

