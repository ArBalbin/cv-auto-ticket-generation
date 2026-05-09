FROM python:3.12-slim

# System deps for opencv-headless and reportlab
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Use headless OpenCV in Docker (no GUI needed)
COPY requirements.txt .
RUN pip install --no-cache-dir \
        opencv-python-headless \
        $(grep -v '^opencv-python$' requirements.txt | grep -v '^#' | grep -v '^$' | tr '\n' ' ')

COPY . .

ENV APP_ENV=production
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

CMD ["python", "app/main.py"]
