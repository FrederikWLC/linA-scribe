import { getPromptPointFromClick } from './prompts.js';

export const pointDragThreshold = 0.25;

export function createPointerSession(mode) {
  return {
    mode,
    actions: [],
    lastProcessed: null,
    hasDragged: false
  };
}

export function capturePointer(event) {
  if (event?.currentTarget?.setPointerCapture) {
    event.currentTarget.setPointerCapture(event.pointerId);
  }
}

export function releasePointer(event) {
  if (event?.currentTarget?.releasePointerCapture) {
    event.currentTarget.releasePointerCapture(event.pointerId);
  }
}

export function shouldProcessPoint(point, mode, pointerSession, threshold = pointDragThreshold) {
  if (mode === 'delete') {
    return true;
  }

  if (!pointerSession?.lastProcessed) {
    return true;
  }

  const dx = point.x - pointerSession.lastProcessed.x;
  const dy = point.y - pointerSession.lastProcessed.y;
  return dx * dx + dy * dy >= threshold * threshold;
}

export function getPointsWithinDistance(event, currentPoints, imageBounds, maxDistance = 3) {
  const clickPoint = getPromptPointFromClick(event, imageBounds);
  if (!clickPoint || currentPoints.length === 0) {
    return [];
  }

  return currentPoints
    .map((point, index) => ({
      point,
      index,
      distanceSq: (point.x - clickPoint.x) ** 2 + (point.y - clickPoint.y) ** 2
    }))
    .filter(({ distanceSq }) => distanceSq <= maxDistance * maxDistance)
    .sort((a, b) => a.distanceSq - b.distanceSq);
}

export function getClosestPointIndex(event, currentPoints, imageBounds, maxDistance = 3) {
  const clickPoint = getPromptPointFromClick(event, imageBounds);
  if (!clickPoint || currentPoints.length === 0) {
    return -1;
  }

  let bestIndex = -1;
  let bestDistance = Infinity;

  currentPoints.forEach((point, index) => {
    const dx = point.x - clickPoint.x;
    const dy = point.y - clickPoint.y;
    const distanceSq = dx * dx + dy * dy;
    if (distanceSq < bestDistance) {
      bestDistance = distanceSq;
      bestIndex = index;
    }
  });

  return bestDistance <= maxDistance * maxDistance ? bestIndex : -1;
}
