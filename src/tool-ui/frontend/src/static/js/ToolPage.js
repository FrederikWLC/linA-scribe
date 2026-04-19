import { derived, get, writable } from 'svelte/store';

const emptyBounds = { left: 0, top: 0, width: 100, height: 100 };

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

export function createToolPageController() {
  const imageUrl = writable('');
  const imageName = writable('');
  const imageSize = writable(0);
  const imageBounds = writable(emptyBounds);
  const pointMode = writable('foreground');
  const points = writable([]);
  const runMessage = writable('');
  const importMessage = writable('Paste, drop, or import an image to begin.');
  const foregroundCount = derived(points, ($points) =>
    $points.filter((point) => point.kind === 'foreground').length
  );
  const backgroundCount = derived(points, ($points) =>
    $points.filter((point) => point.kind === 'background').length
  );

  function loadImageFile(file) {
    if (!file || !file.type.startsWith('image/')) {
      importMessage.set('Please choose an image file.');
      return;
    }

    const previousImageUrl = get(imageUrl);
    if (previousImageUrl) {
      URL.revokeObjectURL(previousImageUrl);
    }

    const nextImageUrl = URL.createObjectURL(file);
    const nextImageName = file.name || 'Pasted image';
    imageUrl.set(nextImageUrl);
    imageName.set(nextImageName);
    imageSize.set(file.size || 0);
    imageBounds.set(emptyBounds);
    points.set([]);
    runMessage.set('');
    importMessage.set(`${nextImageName} loaded.`);
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

  function importFromFiles(event) {
    const [file] = event.currentTarget.files || [];
    loadImageFile(file);
    event.currentTarget.value = '';
  }

  function handleDrop(event) {
    event.preventDefault();
    const [file] = event.dataTransfer.files || [];
    loadImageFile(file);
  }

  function keepDropActive(event) {
    event.preventDefault();
  }

  function placePoint(event) {
    if (!get(imageUrl)) {
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
    points.update(($points) => $points.slice(0, -1));
    runMessage.set('');
  }

  function clearPoints() {
    points.set([]);
    runMessage.set('');
  }

  function runPrompt() {
    if (!get(imageUrl)) {
      runMessage.set('Import an image before running.');
      return;
    }

    const payload = {
      image: {
        name: get(imageName),
        size: get(imageSize),
        previewUrl: get(imageUrl)
      },
      points: get(points).map(({ kind, x, y }) => ({ kind, x, y }))
    };

    console.log('Segmentation prompt payload', payload);
    runMessage.set(
      `Ready to run with ${get(foregroundCount)} foreground and ${get(backgroundCount)} background point(s).`
    );
  }

  function bindToolPageShortcuts() {
    const onKeyDown = (event) => {
      const isUndo = (event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z';
      if (!isUndo || get(points).length === 0) {
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
    };
  }

  return {
    imageUrl,
    imageName,
    imageBounds,
    pointMode,
    points,
    runMessage,
    importMessage,
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
    runPrompt,
    bindToolPageShortcuts
  };
}
