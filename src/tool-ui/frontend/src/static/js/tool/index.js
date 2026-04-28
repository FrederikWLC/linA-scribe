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
  getPointsWithinDistance,
  getClosestPointIndex,
  processPointerEvent,
  updatePreviewBox
} from './pointerSession.js';
import { createSessionHistoryAction, undoAction, undoBoxAction } from './actionHistory.js';
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
  const pointMode = writable('box');
  const points = writable([]);
  const boxes = writable([]);
  const previewBox = writable(null);
  const actionHistory = writable([]);
  const runMessage = writable('');
  const importMessage = writable('Paste, drop, or import an image to begin.');
  let pointerSession = null;
  let runMessageTimeout = null;
  const isImageSet = writable(false);
  const isSettingImage = writable(false);
  let pendingBoxCorner = null;

  function setRunMessage(value, temporary = false) {
    runMessage.set(value);
    if (runMessageTimeout) {
      clearTimeout(runMessageTimeout);
      runMessageTimeout = null;
    }

    if (temporary && value) {
      runMessageTimeout = setTimeout(() => {
        runMessage.set('');
        runMessageTimeout = null;
      }, 3000);
    }
  }

  function setStatusMessage(value) {
    setRunMessage(value, false);
  }

  function setTempStatusMessage(value) {
    setRunMessage(value, true);
  }

  pointMode.subscribe((mode) => {
    if (mode !== 'box') {
      pendingBoxCorner = null;
      previewBox.set(null);
    }
  });

  const selectedModel = derived(selectedModelKey, ($selectedModelKey) =>
    modelOptions.find((model) => model.key === $selectedModelKey) || modelOptions[0]
  );
  const acceptsPrompts = derived(selectedModel, ($selectedModel) => $selectedModel.acceptsPrompts);
  const requiresSetImage = derived(selectedModel, ($selectedModel) => $selectedModel.requiresSetImage);

  const {
    loadImageFile: loadImageFileRaw,
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
    importMessage,
    setStatusMessage,
    setTempStatusMessage
  });

  async function loadImageFile(file) {
    pendingBoxCorner = null;
    previewBox.set(null);
    boxes.set([]);
    return await loadImageFileRaw(file);
  }

  const canRunPredict = derived(
    [imageUrl, isImageSet, isSettingImage, requiresSetImage],
    ([$imageUrl, $isImageSet, $isSettingImage, $requiresSetImage]) =>
      !!$imageUrl &&
      !$isSettingImage &&
      (!$requiresSetImage || $isImageSet)
  );
  const displayedImageUrl = derived(
    [imageUrl, segmentationImageUrl, activeImageMode],
    ([$imageUrl, $segmentationImageUrl, $activeImageMode]) =>
      $activeImageMode === 'segmentation' && $segmentationImageUrl
        ? $segmentationImageUrl
        : $imageUrl
  );
  const isSegmentationActive = derived(
    [segmentationImageUrl, activeImageMode],
    ([$segmentationImageUrl, $activeImageMode]) =>
      $activeImageMode === 'segmentation' && !!$segmentationImageUrl
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
    processPointerEvent({
      event,
      pointerSession,
      pointMode: get(pointMode),
      imageUrl: get(imageUrl),
      acceptsPrompts: get(acceptsPrompts),
      activeImageMode: get(activeImageMode),
      requiresSetImage: get(requiresSetImage),
      isImageSet: get(isImageSet),
      imageBounds: get(imageBounds),
      points: get(points),
      boxes: get(boxes),
      pendingBoxCorner,
      setPendingBoxCorner: (value) => { pendingBoxCorner = value; },
      setPreviewBox: (value) => previewBox.set(value),
      addSessionAction,
      setRunMessage,
      pointsUpdate: (fn) => points.update(fn),
      boxesUpdate: (fn) => boxes.update(fn)
    });
  }

  function beginPointSession(event) {
    pointerSession = createPointerSession(get(pointMode));
    capturePointer(event);
    processPointEvent(event);
  }

  function continuePointSession(event) {
    if (get(pointMode) === 'box' && pendingBoxCorner) {
      updatePreviewBox(event, pendingBoxCorner, get(imageBounds), (value) => previewBox.set(value));
      return;
    }

    if (!pointerSession) {
      return;
    }

    if (pointerSession.mode === 'box') {
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
    boxes.update(($boxes) => undoBoxAction(lastAction, $boxes));
    setRunMessage('');
  }

  function clearPoints() {
    points.set([]);
    boxes.set([]);
    previewBox.set(null);
    actionHistory.set([]);
    pendingBoxCorner = null;
    setRunMessage('');
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
    setRunMessage('');

    if (!get(acceptsPrompts)) {
      pointMode.set('foreground');
    }

    if (get(imageUrl)) {
      await syncModelImageIfNeeded();
    }

    onModelSelected?.(modelKey);
  }

  async function runPrompt() {
    const rawImageUrl = get(imageUrl);
    if (!rawImageUrl) {
      setStatusMessage('Import an image before running.');
      return;
    }

    if (!get(canRunPredict)) {
      setStatusMessage(
        get(isSettingImage)
          ? 'Image is still being set.'
          : 'Set the image before running.'
      );
      return;
    }

    setStatusMessage('Running segmentation...');
    const promptPoints = get(acceptsPrompts) ? get(points) : [];

    try {
      const result = get(requiresSetImage)
        ? await predictSAMwithSetImage(get(selectedModelKey), getAuthHeaders, promptPoints)
        : await predictClassical(get(imageFile), get(selectedModelKey), getAuthHeaders);
      console.log('runPrompt result', {
        model: get(selectedModelKey),
        requiresSetImage: get(requiresSetImage),
        maskBlobSize: result.maskBlob?.size,
      });
      revokeSegmentationImageUrl(get(segmentationImageUrl), get(imageUrl));
      segmentationImageUrl.set(URL.createObjectURL(result.maskBlob));
      activeImageMode.set('segmentation');
      setTempStatusMessage('Segmentation completed.');
    } catch (err) {
      setRunMessage(err instanceof Error ? err.message : String(err));
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
    boxes,
    previewBox,
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
