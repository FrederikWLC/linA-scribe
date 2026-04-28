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

export function isPointInsideBox(point, box) {
  return (
    point.x >= box.x &&
    point.x <= box.x + box.width &&
    point.y >= box.y &&
    point.y <= box.y + box.height
  );
}

export function getBoxesUnderPoint(point, currentBoxes) {
  return currentBoxes
    .map((box, index) => ({ box, index }))
    .filter(({ box }) => isPointInsideBox(point, box));
}

export function getTopBoxUnderPoint(point, currentBoxes) {
  const hits = getBoxesUnderPoint(point, currentBoxes);
  if (hits.length === 0) {
    return null;
  }
  return hits[hits.length - 1];
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

function createBoxFromCorners(first, second) {
  const x1 = Math.min(first.x, second.x);
  const x2 = Math.max(first.x, second.x);
  const y1 = Math.min(first.y, second.y);
  const y2 = Math.max(first.y, second.y);
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`,
    x: x1,
    y: y1,
    width: x2 - x1,
    height: y2 - y1
  };
}

export function updatePreviewBox(event, pendingBoxCorner, imageBounds, setPreviewBox) {
  const point = getPromptPointFromClick(event, imageBounds);
  if (!point) {
    return;
  }

  setPreviewBox(createBoxFromCorners(pendingBoxCorner, point));
}

export function processPointerEvent({
  event,
  pointerSession,
  pointMode,
  imageUrl,
  acceptsPrompts,
  activeImageMode,
  requiresSetImage,
  isImageSet,
  imageBounds,
  points,
  boxes,
  pendingBoxCorner,
  setPendingBoxCorner,
  setPreviewBox,
  addSessionAction,
  setRunMessage,
  pointsUpdate,
  boxesUpdate
}) {
  if (!imageUrl || !acceptsPrompts || activeImageMode !== 'raw' || (requiresSetImage && !isImageSet)) {
    return;
  }

  const mode = pointerSession?.mode || pointMode;

  if (mode === 'delete') {
    const point = getPromptPointFromClick(event, imageBounds);
    const currentBoxes = boxes;
    const boxHits = point ? getBoxesUnderPoint(point, currentBoxes) : [];

    pointsUpdate(($points) => {
      if (pointerSession?.hasDragged) {
        const nearby = getPointsWithinDistance(event, $points, imageBounds);
        const removedPointActions = nearby.map(({ point, index }) => ({ point, index }));
        if (removedPointActions.length > 0) {
          addSessionAction(removedPointActions);
        }

        if (boxHits.length > 0) {
          const removedBoxActions = boxHits.map(({ box, index }) => ({ box, index }));
          addSessionAction(removedBoxActions);
        }

        boxesUpdate(($boxes) =>
          $boxes.filter((_, index) => !boxHits.some((item) => item.index === index))
        );

        return $points.filter((_, index) => !nearby.some((item) => item.index === index));
      }

      const nearby = getPointsWithinDistance(event, $points, imageBounds);
      if (nearby.length > 0) {
        const removedPoints = nearby.map(({ point, index }) => ({ point, index }));
        addSessionAction(removedPoints);
        setRunMessage('');
        return $points.filter((_, index) => !nearby.some((item) => item.index === index));
      }

      if (boxHits.length > 0) {
        const topHit = getTopBoxUnderPoint(point, currentBoxes);
        if (topHit) {
          addSessionAction([{ box: topHit.box, index: topHit.index }]);
          boxesUpdate(($boxes) => $boxes.filter((_, index) => index !== topHit.index));
          setRunMessage('');
        }
      }

      return $points;
    });

    return;
  }

  const point = getPromptPointFromClick(event, imageBounds);
  if (!point || (mode !== 'box' && !shouldProcessPoint(point, mode, pointerSession))) {
    return;
  }

  if (mode === 'box') {
    if (!pendingBoxCorner) {
      setPendingBoxCorner(point);
      setPreviewBox(null);
      return;
    }

    const box = createBoxFromCorners(pendingBoxCorner, point);
    setPendingBoxCorner(null);

    if (!box.width || !box.height) {
      return;
    }

    boxesUpdate(($boxes) => [...$boxes, box]);
    addSessionAction(box);
    setPreviewBox(null);
    return;
  }

  const nextPoint = {
    ...point,
    kind: mode
  };

  pointsUpdate(($points) => {
    addSessionAction(nextPoint);
    pointerSession.lastProcessed = { x: point.x, y: point.y };
    return [...$points, nextPoint];
  });
  setRunMessage('');
}
