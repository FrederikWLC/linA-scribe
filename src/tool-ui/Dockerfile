# Production Dockerfile for Render
# Builds the Svelte frontend and Python FastAPI backend in a single image.

# 1) Build frontend assets
FROM node:22-alpine AS frontend-builder
WORKDIR /frontend

COPY tool-ui/frontend/package*.json ./
RUN npm install

COPY tool-ui/frontend/ ./
RUN npm run build

# 2) Build backend runtime image
FROM python:3.13.9-slim AS runtime
WORKDIR /app

ARG MOBILESAM_CHECKPOINT_URL=""
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODAL_APP_NAME=ScribeSAM
ENV PYTHONPATH=/workspace/src:$PYTHONPATH
ENV MOBILESAM_CHECKPOINT_URL=${MOBILESAM_CHECKPOINT_URL}

COPY tool-ui/backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY tool-ui/backend/app ./app
COPY . /workspace/src

RUN mkdir -p /workspace/src/checkpoints \
    && python - <<'PY'
import os, pathlib, urllib.request, zipfile
url = os.environ.get('MOBILESAM_CHECKPOINT_URL', '').strip()
checkpoint_path = pathlib.Path('/workspace/src/checkpoints/mobile_sam.pt')
if url and not checkpoint_path.exists():
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + '.download')
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, open(tmp_path, 'wb') as out_file:
        out_file.write(response.read())
    if url.lower().endswith('.zip'):
        with zipfile.ZipFile(tmp_path, 'r') as zf:
            found = False
            for name in zf.namelist():
                if pathlib.Path(name).name == checkpoint_path.name:
                    zf.extract(name, checkpoint_path.parent)
                    extracted = checkpoint_path.parent / name
                    extracted.replace(checkpoint_path)
                    found = True
                    break
            if not found:
                raise SystemExit('Checkpoint file not found inside ZIP archive.')
        tmp_path.unlink()
    else:
        tmp_path.replace(checkpoint_path)
PY

COPY --from=frontend-builder /frontend/dist ./frontend/dist

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
