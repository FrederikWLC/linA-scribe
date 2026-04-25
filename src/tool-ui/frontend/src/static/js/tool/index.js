import { derived, get, writable } from 'svelte/store';
import { emptyBounds, calculateImageBoundsFromEvent } from './imageBounds.js';
import { getPromptPointFromClick } from './prompts.js';
import {
  predictClassical,
  predictSAMwithSetImage,
  warmupModels as warmupBackendModels
} from './api.js';
import { modelOptions, resolveInitialModelKey } from './modelConfig.js';
import {
  createPointerSession,
  capturePointer,
  releasePointer,
  shouldProcessPoint,
  getPointsWithinDistance,
  getClosestPointIndex
} from './pointerSession.js';
import { createSessionHistoryAction, undoAction } from './actionHistory.js';
import { bindToolPageShortcuts as bindKeyShortcuts } from './shortcuts.js';
import { createImageLoader, revokeSegmentationImageUrl } from './imageLoader.js';

export function createToolPageController(options = {}) {
  const getAuthHeaders = options.getAuthHeaders || (() => ({}));
  const initialModelKey = options.initialModelKey;
  const onModelSelected = options.onModelSelected;

  const selectedModelKey = writable(resolveInitialModelKey(initialModelKey));
  const imageFile = writable(null);
  const imageUrl = writable('');
  const imageName = writable('');
  const imageSize = writable(0);
  const segmentationImageUrl = writable('');
  const activeImageMode = writable('raw');
  const imageBounds = writable(emptyBounds);
  const pointMode = writable('foreground');
  const points = writable([]);
  const actionHistory = writable([]);
  const runMessage = writable('');
  const importMessage = writable('Paste, drop, or import an image to begin.');
  let pointerSession = null;
  const isImageSet = writable(false);
  const isSettingImage = writable(false);

  const selectedModel = derived(selectedModelKey, ($selectedModelKey) =>
    modelOptions.find((model) => model.key === $selectedModelKey) || modelOptions[0]
  );
  const acceptsPrompts = derived(selectedModel, ($selectedModel) => $selectedModel.acceptsPrompts);
  const requiresSetImage = derived(selectedModel, ($selectedModel) => $selectedModel.requiresSetImage);

  const {
    loadImageFile,
    importFromFiles,
    handleDrop,
    syncModelImageIfNeeded
  } = createImageLoader({
    get,
    getAuthHeaders,
    selectedModelKey,
    requiresSetImage,
    imageFile,
    imageUrl,
    imageName,
    imageSize,
    segmentationImageUrl,
    activeImageMode,
    imageBounds,
    points,
    actionHistory,
    isImageSet,
    isSettingImage,
    runMessage,
    importMessage
  });

  const canRunPredict = derived(
    [imageUrl, isImageSet, isSettingImage, requiresSetImage],
    ([$imageUrl, $isImageSet, $isSettingImage, $requiresSetImage]) =>
      Boolean($imageUrl) &&
      !$isSettingImage &&
      (!$requiresSetImage || $isImageSet)
  );
  const displayedImageUrl = derived(
    [imageUrl, segmentationImageUrl, activeImageMode],
    ([$imageUrl, $segmentationImageUrl, $activeImageMode]) =>
      $activeImageMode === 'segmentation' && $segmentationImageUrl ? $segmentationImageUrl : $imageUrl
  );
  const isSegmentationActive = derived(
    [segmentationImageUrl, activeImageMode],
    ([$segmentationImageUrl, $activeImageMode]) =>
      $activeImageMode === 'segmentation' && Boolean($segmentationImageUrl)
  );
  const foregroundCount = derived(points, ($points) =>
    $points.filter((point) => point.kind === 'foreground').length
  );
  const backgroundCount = derived(points, ($points) =>
    $points.filter((point) => point.kind === 'background').length
  );

  function addSessionAction(action) {
    if (!pointerSession) {
      return;
    }

    if (Array.isArray(action)) {
      pointerSession.actions.push(...action);
    } else {
      pointerSession.actions.push(action);
    }
  }

  function processPointEvent(event) {
    if (
      !get(imageUrl) ||
      !get(acceptsPrompts) ||
      get(activeImageMode) !== 'raw' ||
      (get(requiresSetImage) && !get(isImageSet))
    ) {
      return;
    }

    const mode = pointerSession?.mode || get(pointMode);
    if (mode === 'delete') {
      points.update(($points) => {
        if (pointerSession?.hasDragged) {
          const nearby = getPointsWithinDistance(event, $points, get(imageBounds));
          if (nearby.length === 0) {
            return $points;
          }

          const removedPoints = nearby.map(({ point, index }) => ({ point, index }));
          addSessionAction(removedPoints);

          const keep = $points.filter((_, index) => !nearby.some((item) => item.index === index));
          return keep;
        }

        const indexToRemove = getClosestPointIndex(event, $points, get(imageBounds));
        if (indexToRemove < 0) {
          return $points;
        }

        const removedPoint = $points[indexToRemove];
        addSessionAction({ point: removedPoint, index: indexToRemove });
        pointerSession.lastProcessed = { x: removedPoint.x, y: removedPoint.y };

        return [...$points.slice(0, indexToRemove), ...$points.slice(indexToRemove + 1)];
      });
      runMessage.set('');
      return;
    }

    const point = getPromptPointFromClick(event, get(imageBounds));
    if (!point || !shouldProcessPoint(point, mode, pointerSession)) {
      return;
    }

    const nextPoint = {
      ...point,
      kind: mode
    };

    points.update(($points) => {
      addSessionAction(nextPoint);
      pointerSession.lastProcessed = { x: point.x, y: point.y };
      return [...$points, nextPoint];
    });
    runMessage.set('');
  }

  function beginPointSession(event) {
    pointerSession = createPointerSession(get(pointMode));
    capturePointer(event);
    processPointEvent(event);
  }

  function continuePointSession(event) {
    if (!pointerSession) {
      return;
    }

    pointerSession.hasDragged = true;
    processPointEvent(event);
  }

  function endPointSession(event) {
    releasePointer(event);

    if (!pointerSession || pointerSession.actions.length === 0) {
      pointerSession = null;
      return;
    }

    const sessionAction = createSessionHistoryAction(pointerSession);
    if (sessionAction) {
      actionHistory.update(($history) => [...$history, sessionAction]);
    }

    pointerSession = null;
  }

  function updateImageBounds(event) {
    imageBounds.set(calculateImageBoundsFromEvent(event));
  }

  function placePoint(event) {
    beginPointSession(event);
    endPointSession();
  }

  function undoPoint() {
    if (!get(acceptsPrompts)) {
      return;
    }

    const history = get(actionHistory);
    if (history.length === 0) {
      return;
    }

    const lastAction = history[history.length - 1];
    actionHistory.update(($history) => $history.slice(0, -1));
    points.update(($points) => undoAction(lastAction, $points));
    runMessage.set('');
  }

  function clearPoints() {
    points.set([]);
    actionHistory.set([]);
    runMessage.set('');
  }

  function showRawImage() {
    activeImageMode.set('raw');
  }

  function showSegmentationImage() {
    if (get(segmentationImageUrl)) {
      activeImageMode.set('segmentation');
    }
  }

  async function selectModel(modelKey) {
    if (modelKey === get(selectedModelKey)) {
      return;
    }

    selectedModelKey.set(modelKey);
    revokeSegmentationImageUrl(get(segmentationImageUrl), get(imageUrl));
    segmentationImageUrl.set('');
    activeImageMode.set('raw');
    points.set([]);
    actionHistory.set([]);
    runMessage.set('');

    if (!get(acceptsPrompts)) {
      pointMode.set('foreground');
    }

    if (get(imageUrl)) {
      await syncModelImageIfNeeded();
    }

    if (typeof onModelSelected === 'function') {
      onModelSelected(modelKey);
    }
  }

  async function runPrompt() {
    const rawImageUrl = get(imageUrl);
    if (!rawImageUrl) {
      runMessage.set('Import an image before running.');
      return;
    }

    if (!get(canRunPredict)) {
      runMessage.set(
        get(isSettingImage)
          ? 'Image is still being set.'
          : 'Set the image before running.'
      );
      return;
    }

    runMessage.set('Running segmentation...');
    const promptPoints = get(acceptsPrompts) ? get(points) : [];

    try {
      const maskBlob = get(requiresSetImage)
        ? await predictSAMwithSetImage(get(selectedModelKey), getAuthHeaders, promptPoints)
        : await predictClassical(get(imageFile), get(selectedModelKey), getAuthHeaders);
      revokeSegmentationImageUrl(get(segmentationImageUrl), get(imageUrl));
      segmentationImageUrl.set(URL.createObjectURL(maskBlob));
      activeImageMode.set('segmentation');
      runMessage.set(get(acceptsPrompts)
        ? `Segmentation ready with ${get(foregroundCount)} foreground and ${get(backgroundCount)} background point(s).`
        : 'Segmentation ready.'
      );
    } catch (err) {
      runMessage.set(err instanceof Error ? err.message : String(err));
    }
  }

  function bindToolPageShortcuts() {
    const disposeShortcuts = bindKeyShortcuts({
      acceptsPrompts,
      points,
      loadImageFile,
      undoPoint
    });

    return () => {
      disposeShortcuts();
      const currentImageUrl = get(imageUrl);
      if (currentImageUrl) {
        URL.revokeObjectURL(currentImageUrl);
      }
      revokeSegmentationImageUrl(get(segmentationImageUrl), get(imageUrl));
    };
  }

  return {
    imageUrl,
    imageName,
    modelOptions,
    selectedModelKey,
    selectedModel,
    acceptsPrompts,
    requiresSetImage,
    segmentationImageUrl,
    activeImageMode,
    displayedImageUrl,
    isSegmentationActive,
    imageBounds,
    pointMode,
    points,
    runMessage,
    importMessage,
    isImageSet,
    isSettingImage,
    canRunPredict,
    foregroundCount,
    backgroundCount,
    updateImageBounds,
    importFromFiles,
    handleDrop,
    placePoint,
    beginPointSession,
    continuePointSession,
    endPointSession,
    undoPoint,
    clearPoints,
    showRawImage,
    showSegmentationImage,
    selectModel,
    warmupModels: warmupBackendModels,
    runPrompt,
    bindToolPageShortcuts
  };
}
