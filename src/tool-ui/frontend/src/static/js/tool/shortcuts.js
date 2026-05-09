import { get } from 'svelte/store';

export function bindToolPageShortcuts({ acceptsPrompts, points, boxes, loadImageFile, undoPoint }) {
  const onKeyDown = (event) => {
    const isUndo = (event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z';
    if (!isUndo || !get(acceptsPrompts) || (get(points).length === 0 && get(boxes).length === 0)) {
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
  };
}
