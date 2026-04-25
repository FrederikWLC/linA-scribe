export function createSessionHistoryAction(pointerSession) {
  if (!pointerSession || pointerSession.actions.length === 0) {
    return null;
  }

  return pointerSession.mode === 'delete'
    ? { type: 'removeBatch', actions: pointerSession.actions }
    : { type: 'addBatch', points: pointerSession.actions };
}

export function undoAction(lastAction, currentPoints) {
  if (!lastAction) {
    return currentPoints;
  }

  if (lastAction.type === 'addBatch') {
    const idsToRemove = new Set(lastAction.points.map((point) => point.id));
    return currentPoints.filter((point) => !idsToRemove.has(point.id));
  }

  if (lastAction.type === 'removeBatch') {
    const nextPoints = [...currentPoints];
    const actions = [...lastAction.actions].sort((a, b) => b.index - a.index);
    actions.forEach(({ point, index }) => {
      nextPoints.splice(index, 0, point);
    });
    return nextPoints;
  }

  if (lastAction.type === 'add') {
    return currentPoints.filter((point) => point.id !== lastAction.point.id);
  }

  if (lastAction.type === 'remove') {
    const nextPoints = [...currentPoints];
    nextPoints.splice(lastAction.index, 0, lastAction.point);
    return nextPoints;
  }

  return currentPoints;
}
