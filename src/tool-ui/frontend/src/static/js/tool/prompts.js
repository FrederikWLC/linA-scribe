import { clampPercent } from './imageBounds.js';

function makePointId() {
  return crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;
}

export function buildAutoseedPoints(data) {
  const seededPrompts = data?.autoseed_prompts || data?.set_image?.autoseed_prompts || [];
  const width = Number(data?.width);
  const height = Number(data?.height);

  if (!Array.isArray(seededPrompts) || !width || !height) {
    return [];
  }

  return seededPrompts
    .map((prompt) => {
      const x = Number(prompt?.x);
      const y = Number(prompt?.y);
      const label = Number(prompt?.label);

      if (!Number.isFinite(x) || !Number.isFinite(y)) {
        return null;
      }

      return {
        id: makePointId(),
        kind: label === 0 ? 'background' : 'foreground',
        x: clampPercent((x / Math.max(width - 1, 1)) * 100),
        y: clampPercent((y / Math.max(height - 1, 1)) * 100)
      };
    })
    .filter(Boolean);
}

export function getPromptPointFromClick(event, imageBounds) {
  if (!imageBounds?.width || !imageBounds?.height) {
    return null;
  }

  const stageRect = event.currentTarget.getBoundingClientRect();
  if (!stageRect.width || !stageRect.height) {
    return null;
  }

  const stageX = ((event.clientX - stageRect.left) / stageRect.width) * 100;
  const stageY = ((event.clientY - stageRect.top) / stageRect.height) * 100;
  const rawX = ((stageX - imageBounds.left) / imageBounds.width) * 100;
  const rawY = ((stageY - imageBounds.top) / imageBounds.height) * 100;

  if (rawX < 0 || rawX > 100 || rawY < 0 || rawY > 100) {
    return null;
  }

  const x = clampPercent(rawX);
  const y = clampPercent(rawY);

  return { id: makePointId(), x, y };
}
