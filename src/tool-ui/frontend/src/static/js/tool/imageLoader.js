import { setBackendImage } from './api.js';
import { buildAutoseedPoints } from './prompts.js';

export function revokeSegmentationImageUrl(previousSegmentationImageUrl, currentImageUrl) {
  if (previousSegmentationImageUrl && previousSegmentationImageUrl !== currentImageUrl) {
    URL.revokeObjectURL(previousSegmentationImageUrl);
  }
}

async function uploadAndSeedImage(file, selectedModelKey, getAuthHeaders, requiresSetImage) {
  if (!requiresSetImage) {
    return [];
  }

  const data = await setBackendImage(file, selectedModelKey, getAuthHeaders);
  return buildAutoseedPoints(data);
}

export function createImageLoader(options) {
  const {
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
    setStatusMessage
  } = options;

  async function loadImageFile(file) {
    if (!file || !file.type.startsWith('image/')) {
      importMessage.set('Please choose an image file.');
      return;
    }

    const previousImageUrl = get(imageUrl);
    if (previousImageUrl) {
      URL.revokeObjectURL(previousImageUrl);
    }
    revokeSegmentationImageUrl(get(segmentationImageUrl), get(imageUrl));

    const nextImageUrl = URL.createObjectURL(file);
    const nextImageName = file.name || 'Pasted image';
    imageFile.set(file);
    imageUrl.set(nextImageUrl);
    imageName.set(nextImageName);
    imageSize.set(file.size || 0);
    segmentationImageUrl.set('');
    activeImageMode.set('raw');
    imageBounds.set({ left: 0, top: 0, width: 0, height: 0 });
    points.set([]);
    actionHistory.set([]);
    isImageSet.set(false);
    isSettingImage.set(get(requiresSetImage));
    runMessage.set('');
    importMessage.set(
      get(requiresSetImage)
        ? `${nextImageName} loaded. Setting model image...`
        : `${nextImageName} loaded.`
    );

    if (!get(requiresSetImage)) {
      return;
    }

    try {
      if (typeof setStatusMessage === 'function') {
        setStatusMessage('Setting model image...', false);
      }

      const nextPoints = await uploadAndSeedImage(
        file,
        get(selectedModelKey),
        getAuthHeaders,
        get(requiresSetImage)
      );
      points.set(nextPoints);
      actionHistory.set([]);
      isImageSet.set(get(requiresSetImage) ? true : false);
      importMessage.set(
        nextPoints.length > 0
          ? `${nextImageName} loaded with ${nextPoints.length} seed point(s).`
          : `${nextImageName} loaded.`
      );
      if (typeof setStatusMessage === 'function') {
        setStatusMessage(
          nextPoints.length > 0
            ? `${nextImageName} loaded with ${nextPoints.length} seed point(s).`
            : `${nextImageName} loaded.`,
          true
        );
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      importMessage.set(message);
      if (typeof setStatusMessage === 'function') {
        setStatusMessage(message);
      }
    } finally {
      isSettingImage.set(false);
    }
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

  async function syncModelImageIfNeeded() {
    const file = get(imageFile);
    if (!file || !get(requiresSetImage)) {
      isImageSet.set(false);
      isSettingImage.set(false);
      return;
    }

    isImageSet.set(false);
    isSettingImage.set(true);
    const statusName = get(imageName) || 'Image';
    importMessage.set(`${statusName} loaded. Setting model image...`);
    if (typeof setStatusMessage === 'function') {
      setStatusMessage(`Setting model image...`, { autoClear: false });
    }

    try {
      const nextPoints = await uploadAndSeedImage(
        file,
        get(selectedModelKey),
        getAuthHeaders,
        get(requiresSetImage)
      );
      points.set(nextPoints);
      actionHistory.set([]);
      isImageSet.set(get(requiresSetImage) ? true : false);
      const completedMessage =
        nextPoints.length > 0
          ? `${statusName} loaded with ${nextPoints.length} seed point(s).`
          : `${statusName} loaded.`;
      importMessage.set(completedMessage);
      if (typeof setStatusMessage === 'function') {
        setStatusMessage(completedMessage, true);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      importMessage.set(message);
      if (typeof setStatusMessage === 'function') {
        setStatusMessage(message, true);
      }
    } finally {
      isSettingImage.set(false);
    }
  }

  return {
    loadImageFile,
    importFromFiles,
    handleDrop,
    syncModelImageIfNeeded
  };
}
