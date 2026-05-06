import { readErrorMessage } from './utils.js';
import { buildBoxPayloads } from './prompts.js';

export async function setBackendImage(file, selectedModelID, getAuthHeaders) {
  const formData = new FormData();
  formData.append('file', file);
  const params = new URLSearchParams({ model: selectedModelID });

  const response = await fetch(`/api/scribe/set-image-for-sam?${params.toString()}`, {
    method: 'POST',
    headers: getAuthHeaders(),
    body: formData
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response, 'Failed to set image'));
  }

  return response.json();
}

const SAM_MODEL_IDENTIFIER = 'sam';
const CLASSICAL_MODEL_IDENTIFIER = 'gaussian';

function isSAMModel(selectedModelID) {
  return SAM_MODEL_IDENTIFIER === selectedModelID;
}

function isClassicalModel(selectedModelID) {
  return CLASSICAL_MODEL_IDENTIFIER === selectedModelID;
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

export async function predictWithSAM(selectedModelID, getAuthHeaders, promptPoints, boxes = []) {
  if (!isSAMModel(selectedModelID)) {
    throw new Error('predictWithSAM is only supported for SAM models.');
  }

  const response = await fetch('/api/scribe/predict-with-sam', {
    method: 'POST',
    headers: {
      ...getAuthHeaders(),
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model: selectedModelID,
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

export async function predictWithClassical(imageFile, selectedModelID, getAuthHeaders) {
  if (!isClassicalModel(selectedModelID)) {
    throw new Error('predictWithClassical is only supported for classical models.');
  }

  const params = new URLSearchParams({ model: selectedModelID });
  const formData = new FormData();
  formData.append('file', imageFile);

  const response = await fetch(`/api/scribe/predict-with-classical?${params.toString()}`, {
    method: 'POST',
    headers: getAuthHeaders(),
    body: formData
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response, 'Segmentation failed'));
  }

  return parsePredictResponse(response);
}

export async function warmupSAMForUser(getAuthHeaders) {
  try {
    await fetch('/api/scribe/warmup-sam-for-user', {
      method: 'POST',
      headers: getAuthHeaders()
    });
  } catch (err) {
    console.warn('Model warmup failed', err);
  }
}

