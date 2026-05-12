import * as twemoji from 'twemoji';
import { derived, get, writable } from 'svelte/store';
import { emptyBounds, calculateImageBoundsFromEvent } from './imageBounds.js';
import {
  predictWithClassical,
  predictWithSAM,
  warmupSAMForUser as warmupSAMForUserAPI
} from './api.js';
import { modelOptions } from './modelConfig.js';
import {
  createPointerSession,
  capturePointer,
  releasePointer,
  processPointerEvent,
  updatePreviewBox
} from './prompts/pointerSession.js';
import { createSessionHistoryAction, undoAction, addSessionAction } from './prompts/actionHistory.js';
import { bindToolPageShortcuts as bindKeyShortcuts } from './shortcuts.js';
import { createImageLoader, revokeSegmentationImageUrl } from './imageLoader.js';

const twemojiParser = twemoji.default ?? twemoji;
const twemojiBase = 'https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg';

export const emojiUrl = (emoji) => { // for stable emoji icons in UI, independt of platform support
  const codePoint = twemojiParser.convert.toCodePoint(emoji);
  return `${twemojiBase}/${codePoint}.svg`;
};

export function createToolPageController(options = {}) {
  const getAuthHeaders = options.getAuthHeaders || (() => ({}));
  const onModelSelected = options.onModelSelected;

  const selectedModelID = writable(null);
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
  const isDecodingMask = writable(false);
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

  const selectedModel = derived(selectedModelID, ($selectedModelID) =>
    modelOptions.find((model) => model.id === $selectedModelID) || null
  );
  const acceptsPrompts = derived(selectedModel, ($selectedModel) => $selectedModel?.acceptsPrompts ?? false);
  const requiresSetImage = derived(selectedModel, ($selectedModel) => $selectedModel?.requiresSetImage ?? false);

  const {
    loadImageFile: loadImageFileRaw,
    importFromFiles: importFromFilesRaw,
    handleDrop: handleDropRaw,
    syncModelImageIfNeeded
  } = createImageLoader({
    get,
    getAuthHeaders,
    selectedModelID,
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

  function clearPrompts() {
    points.set([]);
    boxes.set([]);
    previewBox.set(null);
    actionHistory.set([]);
    pendingBoxCorner = null;
    setRunMessage('');
  }


  async function loadImageFile(file) {
    clearPrompts();
    return await loadImageFileRaw(file);
  }

  async function importFromFiles(event) {
    clearPrompts();
    return await importFromFilesRaw(event);
  }

  async function handleDrop(event) {
    clearPrompts();
    return await handleDropRaw(event);
  }

  const canRunPredict = derived(
    [imageUrl, isImageSet, isSettingImage, isDecodingMask, requiresSetImage, selectedModel],
    ([$imageUrl, $isImageSet, $isSettingImage, $isDecodingMask, $requiresSetImage, $selectedModel]) =>
      !!$imageUrl &&
      !!$selectedModel &&
      !$isSettingImage &&
      !$isDecodingMask &&
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
    $points.filter((point) => point && point.kind === 'foreground').length
  );
  const backgroundCount = derived(points, ($points) =>
    $points.filter((point) => point && point.kind === 'background').length
  );

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
      addSessionAction: (action) => addSessionAction(pointerSession, action),
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

    if (!pointerSession || pointerSession.mode !== 'delete') {
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
    const nextPrompts = undoAction(lastAction, {
      points: get(points),
      boxes: get(boxes)
    });
    points.set(nextPrompts.points);
    boxes.set(nextPrompts.boxes);
    setRunMessage('');
  }

  function clearPoints() {
    clearPrompts();
  }

  function showRawImage() {
    activeImageMode.set('raw');
  }

  function showSegmentationImage() {
    if (get(segmentationImageUrl)) {
      activeImageMode.set('segmentation');
    }
  }

  async function selectModel(modelID) {
    if (modelID === get(selectedModelID)) {
      return;
    }

    selectedModelID.set(modelID);
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

    onModelSelected?.(modelID);
  }

  async function warmupSAMForUser() {
    return warmupSAMForUserAPI(getAuthHeaders);
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
          : get(isDecodingMask)
          ? 'Mask is still decoding.'
          : 'Set the image before running.'
      );
      return;
    }

    isDecodingMask.set(true);
    setRunMessage('Decoding mask...');
    const promptPoints = get(acceptsPrompts) ? get(points) : [];
    const currentBoxes = get(acceptsPrompts) ? get(boxes) : [];

    try {
      const result = get(requiresSetImage)
        ? await predictWithSAM(get(selectedModelID), getAuthHeaders, promptPoints, currentBoxes)
        : await predictWithClassical(get(imageFile), get(selectedModelID), getAuthHeaders);
      console.log('runPrompt result', {
        model: get(selectedModelID),
        requiresSetImage: get(requiresSetImage),
        maskBlobSize: result.maskBlob?.size,
      });
      revokeSegmentationImageUrl(get(segmentationImageUrl), get(imageUrl));
      segmentationImageUrl.set(URL.createObjectURL(result.maskBlob));
      activeImageMode.set('segmentation');
      setTempStatusMessage('Segmentation completed.');
    } catch (err) {
      setRunMessage(err instanceof Error ? err.message : String(err));
    } finally {
      isDecodingMask.set(false);
    }
  }

  function bindToolPageShortcuts() {
    const disposeShortcuts = bindKeyShortcuts({
      acceptsPrompts,
      points,
      boxes,
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
    selectedModelID,
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
    isDecodingMask,
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
    warmupSAMForUser,
    runPrompt,
    bindToolPageShortcuts
  };
}
