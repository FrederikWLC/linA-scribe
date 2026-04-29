import { readErrorMessage } from './utils.js';
import { buildBoxPayloads } from './prompts.js';

export async function setBackendImage(file, selectedModelKey, getAuthHeaders) {
  const formData = new FormData();
  formData.append('file', file);
  const params = new URLSearchParams({ model: selectedModelKey });

  const response = await fetch(`/api/scribe/set-image?${params.toString()}`, {
    method: 'POST',
    headers: getAuthHeaders(),
    body: formData
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response, 'Failed to set image'));
  }

  return response.json();
}

const SAM_MODEL_KEYS = ['modal-mobilesam'];
const CLASSICAL_MODEL_KEYS = ['gaussian'];

function isSAMModel(selectedModelKey) {
  return SAM_MODEL_KEYS.includes(selectedModelKey);
}

function isClassicalModel(selectedModelKey) {
  return CLASSICAL_MODEL_KEYS.includes(selectedModelKey);
}

function base64ToBlob(base64, type = 'image/png') {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i += 1) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return new Blob([bytes], { type });
}

async function parsePredictResponse(response) {
  const data = await response.json();
  return {
    maskBlob: base64ToBlob(data.mask_png, 'image/png'),
  };
}

export async function predictSAMwithSetImage(selectedModelKey, getAuthHeaders, promptPoints, boxes = []) {
  if (!isSAMModel(selectedModelKey)) {
    throw new Error('predictSAMwithSetImage is only supported for SAM models.');
  }

  const response = await fetch('/api/scribe/predict-set-image', {
    method: 'POST',
    headers: {
      ...getAuthHeaders(),
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model: selectedModelKey,
      coordinate_space: 'percent',
      x: promptPoints.map((point) => point.x),
      y: promptPoints.map((point) => point.y),
      labels: promptPoints.map((point) => (point.kind === 'foreground' ? 1 : 0)),
      x1s: boxes.map((box) => box.x1),
      y1s: boxes.map((box) => box.y1),
      x2s: boxes.map((box) => box.x2),
      y2s: boxes.map((box) => box.y2)
    })
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response, 'Segmentation failed'));
  }

  return parsePredictResponse(response);
}

export async function predictClassical(imageFile, selectedModelKey, getAuthHeaders) {
  if (!isClassicalModel(selectedModelKey)) {
    throw new Error('predictClassical is only supported for classical models.');
  }

  const params = new URLSearchParams({ model: selectedModelKey });
  const formData = new FormData();
  formData.append('file', imageFile);

  const response = await fetch(`/api/scribe/predict?${params.toString()}`, {
    method: 'POST',
    headers: getAuthHeaders(),
    body: formData
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response, 'Segmentation failed'));
  }

  return parsePredictResponse(response);
}

export async function warmupModels(getAuthHeaders) {
  try {
    await fetch('/api/scribe/warmup', {
      method: 'POST',
      headers: getAuthHeaders()
    });
  } catch (err) {
    console.warn('Model warmup failed', err);
  }
}

