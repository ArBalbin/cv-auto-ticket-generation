FROM python:3.12-slim

# System deps for opencv-headless and reportlab
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements.txt is the server set: it already pins opencv-python-headless
# and deliberately excludes ultralytics, so no grep-and-substitute is needed
# here any more. The camera machine uses requirements-detector.txt instead.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the face model pack into the image.
#
# FaceAnalysis downloads buffalo_s.zip on first use, into ~/.insightface —
# a path that does not survive a container restart. On a host that sleeps,
# that meant every cold start re-downloaded 158MB before the first face
# request could be answered, and the request that triggered it paid the wait.
#
# Only the two modules face_service actually uses are kept. Deleting the rest
# takes 143MB out of the image; see the comment in app/services/face_service.py
# for why they are not needed and the measurements behind it.
#
# This runs before `COPY . .` so editing application code does not invalidate
# the layer and re-download the pack on every build.
ARG FACE_MODEL_PACK=buffalo_s
RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='${FACE_MODEL_PACK}', allowed_modules=['detection','recognition'], providers=['CPUExecutionProvider'])" \
    && rm -f "/root/.insightface/models/${FACE_MODEL_PACK}/1k3d68.onnx" \
             "/root/.insightface/models/${FACE_MODEL_PACK}/2d106det.onnx" \
             "/root/.insightface/models/${FACE_MODEL_PACK}/genderage.onnx"

COPY . .

ENV APP_ENV=production
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

CMD ["python", "app/main.py"]
