import { clampPercent } from './imageBounds.js';

function makePointId() {
  return crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;
}

export function createBoxFromCorners(first, second) {
  const x1 = Math.min(first.x, second.x);
  const x2 = Math.max(first.x, second.x);
  const y1 = Math.min(first.y, second.y);
  const y2 = Math.max(first.y, second.y);
  const width = x2 - x1;
  const height = y2 - y1;
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`,
    x1,
    y1,
    x2,
    y2,
    x: x1,
    y: y1,
    width,
    height,
  };
}

export function buildBoxPayloads(boxes) {
  return boxes.map((box) => {
    if (box.x1 != null && box.y1 != null && box.x2 != null && box.y2 != null) {
      return {
        x1: box.x1,
        y1: box.y1,
        x2: box.x2,
        y2: box.y2,
      };
    }

    if (box.x != null && box.y != null && box.width != null && box.height != null) {
      return {
        x1: box.x,
        y1: box.y,
        x2: box.x + box.width,
        y2: box.y + box.height,
      };
    }

    throw new Error('Box prompts must include x1, y1, x2, y2');
  });
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
