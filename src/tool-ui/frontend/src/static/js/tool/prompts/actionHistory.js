export function addSessionAction(pointerSession, action) {
    if (!pointerSession) {
      return;
    }

    if (Array.isArray(action)) {
      pointerSession.actions.push(...action.filter(Boolean));
    } else if (action) {
      pointerSession.actions.push(action);
    }
  }

export function createSessionHistoryAction(pointerSession) {
  if (!pointerSession || pointerSession.actions.length === 0) {
    return null;
  }

  if (pointerSession.mode === 'delete') {
    return { type: 'removeBatch', actions: pointerSession.actions };
  }

  if (pointerSession.mode === 'box') {
    return { type: 'box', box: pointerSession.actions.find(Boolean) };
  }

  const pointActions = pointerSession.actions.filter(Boolean);
  if (pointActions.length === 1) {
    return { type: 'point', point: pointActions[0] };
  }

  return { type: 'pointBatch', points: pointActions };
}

export function undoAction(lastAction, currentPrompts) {
  const validPoints = currentPrompts.points.filter(Boolean);
  const validBoxes = currentPrompts.boxes.filter(Boolean);

  if (!lastAction) {
    return { points: validPoints, boxes: validBoxes };
  }

  if (lastAction.type === 'point') { // if point action
    const idToRemove = lastAction.point?.id;
    return {
      points: validPoints.filter((point) => point.id !== idToRemove),
      boxes: validBoxes
    };
  } // remove the point

  if (lastAction.type === 'pointBatch') {
    const idsToRemove = new Set(lastAction.points.map((point) => point?.id).filter(Boolean));
    return {
      points: validPoints.filter((point) => !idsToRemove.has(point.id)),
      boxes: validBoxes
    };
  }

  if (lastAction.type === 'box') {
    const idToRemove = lastAction.box?.id;
    return {
      points: validPoints,
      boxes: validBoxes.filter((box) => box.id !== idToRemove)
    };
  }

  if (lastAction.type === 'removeBatch') { // if removeBatch action
    const nextPoints = [...validPoints]; // makes a copy
    const pointActions = [...lastAction.actions] // makes a copy
      .filter((action) => action.point) // filtering out non-point actions
      .sort((a, b) => b.index - a.index); // reverse sort to restore at original indices
    pointActions.forEach(({ point, index }) => {
      nextPoints.splice(index, 0, point); // at index, remove 0 (nothing), and insert point
    });


    // and do the same thing for boxes
    const nextBoxes = [...validBoxes]; 
    const boxActions = [...lastAction.actions]
      .filter((action) => action.box) // filtering out non-box actions
      .sort((a, b) => b.index - a.index);
    boxActions.forEach(({ box, index }) => {
      nextBoxes.splice(index, 0, box);
    });
    return { points: nextPoints, boxes: nextBoxes };
  }

  // if no action history found, we return current (empty) prompts as is
  return { points: validPoints, boxes: validBoxes };
}
