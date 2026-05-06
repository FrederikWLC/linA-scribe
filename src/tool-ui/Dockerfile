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

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODAL_APP_NAME=ScribeSAM
ENV PYTHONPATH=/workspace/src:$PYTHONPATH

COPY tool-ui/backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY tool-ui/backend/app ./app
COPY . /workspace/src

RUN mkdir -p /workspace/src/checkpoints && python - <<'PY'
import pathlib
import urllib.request

urls = {
    'mobile_sam.pt': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/mobile_sam.pt',
    'sam_vit_h_4b8939.pth': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
    'sam2.1_hiera_large.pt': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt',
}
root = pathlib.Path('/workspace/src/checkpoints')
root.mkdir(parents=True, exist_ok=True)
for filename, url in urls.items():
    dest = root / filename
    if not dest.exists():
        print(f'Downloading {filename} from {url}')
        urllib.request.urlretrieve(url, dest)
PY

COPY --from=frontend-builder /frontend/dist ./frontend/dist

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
