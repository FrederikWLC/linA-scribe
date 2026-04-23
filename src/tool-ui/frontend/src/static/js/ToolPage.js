import { derived, get, writable } from 'svelte/store';

const emptyBounds = { left: 0, top: 0, width: 100, height: 100 };
const modelOptions = [
  {
    key: 'modal-mobilesam',
    label: 'BestMobileSAMv2Implementation',
    requiresSetImage: true,
    acceptsPrompts: true
  },
  { key: 'canny-fill', label: 'CannyFill', requiresSetImage: false, acceptsPrompts: false },
  { key: 'gaussian', label: 'Gaussian', requiresSetImage: false, acceptsPrompts: false },
  { key: 'grabcut-auto-brush', label: 'GC+brush', requiresSetImage: false, acceptsPrompts: false },
  { key: 'otsu', label: 'Otsu', requiresSetImage: false, acceptsPrompts: false },
  { key: 'modal-gfsam', label: 'GFSAM', requiresSetImage: true, acceptsPrompts: false },
  { key: 'modal-fatesam2d', label: 'FATESAM2D', requiresSetImage: true, acceptsPrompts: true }
];

function makePointId() {
  if (crypto.randomUUID) {
    return crypto.randomUUID();
  }

  return `${Date.now()}-${Math.random()}`;
}

function imagePointStyle(point, bounds) {
  const left = bounds.left + (point.x / 100) * bounds.width;
  const top = bounds.top + (point.y / 100) * bounds.height;
  return `left: ${left}%; top: ${top}%;`;
}

function clampPercent(value) {
  return Math.min(100, Math.max(0, value));
}

async function readErrorMessage(response, fallback) {
  const text = await response.text();
  if (!text) {
    return fallback;
  }

  try {
    const data = JSON.parse(text);
    return data.detail || data.message || fallback;
  } catch {
    return text;
  }
}

export function createToolPageController(options = {}) {
  const getAuthHeaders = options.getAuthHeaders || (() => ({}));
  const selectedModelKey = writable('modal-mobilesam');
  const imageFile = writable(null);
  const imageUrl = writable('');
  const imageName = writable('');
  const imageSize = writable(0);
  const segmentationImageUrl = writable('');
  const activeImageMode = writable('raw');
  const imageBounds = writable(emptyBounds);
  const pointMode = writable('foreground');
  const points = writable([]);
  const runMessage = writable('');
  const importMessage = writable('Paste, drop, or import an image to begin.');
  const isImageSet = writable(false);
  const isSettingImage = writable(false);
  const selectedModel = derived(selectedModelKey, ($selectedModelKey) =>
    modelOptions.find((model) => model.key === $selectedModelKey) || modelOptions[0]
  );
  const acceptsPrompts = derived(selectedModel, ($selectedModel) => $selectedModel.acceptsPrompts);
  const requiresSetImage = derived(selectedModel, ($selectedModel) => $selectedModel.requiresSetImage);
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

  function revokeSegmentationImageUrl() {
    const previousSegmentationImageUrl = get(segmentationImageUrl);
    if (previousSegmentationImageUrl && previousSegmentationImageUrl !== get(imageUrl)) {
      URL.revokeObjectURL(previousSegmentationImageUrl);
    }
  }

  async function setBackendImage(file) {
    const formData = new FormData();
    formData.append('file', file);
    const params = new URLSearchParams({ model: get(selectedModelKey) });

    const response = await fetch(`/api/scribe/set-image?${params.toString()}`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: formData
    });
    if (!response.ok) {
      throw new Error(await readErrorMessage(response, 'Failed to set image'));
    }

    return response.json();
  }

  async function warmupModels() {
    try {
      await fetch('/api/scribe/warmup', {
        method: 'POST',
        headers: getAuthHeaders()
      });
    } catch (err) {
      console.warn('Model warmup failed', err);
    }
  }

  function applyAutoseedPrompts(data) {
    if (!get(acceptsPrompts)) {
      return 0;
    }

    const seededPrompts = data?.autoseed_prompts || data?.set_image?.autoseed_prompts || [];
    const width = Number(data?.width);
    const height = Number(data?.height);

    if (!Array.isArray(seededPrompts) || !Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
      return 0;
    }

    const nextPoints = seededPrompts
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

    points.set(nextPoints);
    return nextPoints.length;
  }

  async function loadImageFile(file) {
    if (!file || !file.type.startsWith('image/')) {
      importMessage.set('Please choose an image file.');
      return;
    }

    const previousImageUrl = get(imageUrl);
    if (previousImageUrl) {
      URL.revokeObjectURL(previousImageUrl);
    }
    revokeSegmentationImageUrl();

    const nextImageUrl = URL.createObjectURL(file);
    const nextImageName = file.name || 'Pasted image';
    imageFile.set(file);
    imageUrl.set(nextImageUrl);
    imageName.set(nextImageName);
    imageSize.set(file.size || 0);
    segmentationImageUrl.set('');
    activeImageMode.set('raw');
    imageBounds.set(emptyBounds);
    points.set([]);
    isImageSet.set(false);
    isSettingImage.set(get(requiresSetImage));
    runMessage.set('');
    importMessage.set(
      get(requiresSetImage) ? `${nextImageName} loaded. Setting model image...` : `${nextImageName} loaded.`
    );

    if (!get(requiresSetImage)) {
      return;
    }

    try {
      const data = await setBackendImage(file);
      const seededCount = applyAutoseedPrompts(data);
      isImageSet.set(true);
      importMessage.set(
        seededCount > 0 ? `${nextImageName} loaded with ${seededCount} seed point(s).` : `${nextImageName} loaded.`
      );
    } catch (err) {
      importMessage.set(err instanceof Error ? err.message : String(err));
    } finally {
      isSettingImage.set(false);
    }
  }

  function updateImageBounds(event) {
    const img = event.currentTarget;
    const naturalRatio = img.naturalWidth / img.naturalHeight;
    const stageRatio = img.clientWidth / img.clientHeight;

    if (!Number.isFinite(naturalRatio) || !Number.isFinite(stageRatio)) {
      imageBounds.set(emptyBounds);
      return;
    }

    if (stageRatio > naturalRatio) {
      const width = (naturalRatio / stageRatio) * 100;
      imageBounds.set({ left: (100 - width) / 2, top: 0, width, height: 100 });
      return;
    }

    const height = (stageRatio / naturalRatio) * 100;
    imageBounds.set({ left: 0, top: (100 - height) / 2, width: 100, height });
  }

  function openFilePicker(fileInput) {
    fileInput?.click();
  }

  async function importFromFiles(event) {
    const [file] = event.currentTarget.files || [];
    await loadImageFile(file);
    event.currentTarget.value = '';
  }

  async function handleDrop(event) {
    event.preventDefault();
    const [file] = event.dataTransfer.files || [];
    await loadImageFile(file);
  }

  function keepDropActive(event) {
    event.preventDefault();
  }

  function placePoint(event) {
    if (!get(imageUrl) || !get(acceptsPrompts)) {
      return;
    }

    if (get(activeImageMode) !== 'raw') {
      return;
    }

    const bounds = event.currentTarget.getBoundingClientRect();
    const stageX = ((event.clientX - bounds.left) / bounds.width) * 100;
    const stageY = ((event.clientY - bounds.top) / bounds.height) * 100;
    const currentImageBounds = get(imageBounds);
    const imageX = ((stageX - currentImageBounds.left) / currentImageBounds.width) * 100;
    const imageY = ((stageY - currentImageBounds.top) / currentImageBounds.height) * 100;

    if (imageX < 0 || imageX > 100 || imageY < 0 || imageY > 100) {
      return;
    }

    points.update(($points) => [
      ...$points,
      {
        id: makePointId(),
        kind: get(pointMode),
        x: imageX,
        y: imageY
      }
    ]);
    runMessage.set('');
  }

  function undoPoint() {
    if (!get(acceptsPrompts)) {
      return;
    }

    points.update(($points) => $points.slice(0, -1));
    runMessage.set('');
  }

  function clearPoints() {
    points.set([]);
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

  async function syncModelImageIfNeeded() {
    const file = get(imageFile);
    if (!file || !get(requiresSetImage)) {
      isImageSet.set(false);
      isSettingImage.set(false);
      return;
    }

    isImageSet.set(false);
    isSettingImage.set(true);
    importMessage.set(`${get(imageName) || 'Image'} loaded. Setting model image...`);

    try {
      const data = await setBackendImage(file);
      const seededCount = applyAutoseedPrompts(data);
      isImageSet.set(true);
      importMessage.set(
        seededCount > 0
          ? `${get(imageName) || 'Image'} loaded with ${seededCount} seed point(s).`
          : `${get(imageName) || 'Image'} loaded.`
      );
    } catch (err) {
      importMessage.set(err instanceof Error ? err.message : String(err));
    } finally {
      isSettingImage.set(false);
    }
  }

  async function selectModel(modelKey) {
    if (modelKey === get(selectedModelKey)) {
      return;
    }

    selectedModelKey.set(modelKey);
    revokeSegmentationImageUrl();
    segmentationImageUrl.set('');
    activeImageMode.set('raw');
    points.set([]);
    runMessage.set('');

    if (!get(acceptsPrompts)) {
      pointMode.set('foreground');
    }

    if (get(imageUrl)) {
      await syncModelImageIfNeeded();
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

    const payload = {
      image: {
        name: get(imageName),
        size: get(imageSize),
        previewUrl: get(imageUrl)
      },
      model: get(selectedModelKey),
      points: get(acceptsPrompts) ? get(points).map(({ kind, x, y }) => ({ kind, x, y })) : []
    };

    console.log('Segmentation prompt payload', payload);
    runMessage.set('Running segmentation...');

    const promptPoints = get(acceptsPrompts) ? get(points) : [];
    const params = new URLSearchParams({ coordinate_space: 'percent' });
    for (const point of promptPoints) {
      params.append('x', String(point.x));
      params.append('y', String(point.y));
      params.append('labels', point.kind === 'foreground' ? '1' : '0');
    }
    params.set('model', get(selectedModelKey));

    try {
      const request = get(requiresSetImage)
        ? fetch('/api/scribe/predict-set-image', {
            method: 'POST',
            headers: {
              ...getAuthHeaders(),
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              model: get(selectedModelKey),
              coordinate_space: 'percent',
              x: promptPoints.map((point) => point.x),
              y: promptPoints.map((point) => point.y),
              labels: promptPoints.map((point) => (point.kind === 'foreground' ? 1 : 0))
            })
          })
        : fetch(`/api/scribe/predict?${params.toString()}`, {
            method: 'POST',
            headers: getAuthHeaders(),
            body: (() => {
              const formData = new FormData();
              formData.append('file', get(imageFile));
              return formData;
            })()
          });
      const response = await request;
      if (!response.ok) {
        throw new Error(await readErrorMessage(response, 'Segmentation failed'));
      }

      const maskBlob = await response.blob();
      revokeSegmentationImageUrl();
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
    const onKeyDown = (event) => {
      const isUndo = (event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z';
      if (!isUndo || !get(acceptsPrompts) || get(points).length === 0) {
        return;
      }

      event.preventDefault();
      undoPoint();
    };

    const onPaste = (event) => {
      const items = Array.from(event.clipboardData?.items || []);
      const imageItem = items.find((item) => item.type.startsWith('image/'));
      const file = imageItem?.getAsFile();

      if (!file) {
        return;
      }

      event.preventDefault();
      loadImageFile(file);
    };

    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('paste', onPaste);

    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('paste', onPaste);
      const currentImageUrl = get(imageUrl);
      if (currentImageUrl) {
        URL.revokeObjectURL(currentImageUrl);
      }
      revokeSegmentationImageUrl();
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
    imagePointStyle,
    updateImageBounds,
    openFilePicker,
    importFromFiles,
    handleDrop,
    keepDropActive,
    placePoint,
    undoPoint,
    clearPoints,
    showRawImage,
    showSegmentationImage,
    selectModel,
    warmupModels,
    runPrompt,
    bindToolPageShortcuts
  };
}
