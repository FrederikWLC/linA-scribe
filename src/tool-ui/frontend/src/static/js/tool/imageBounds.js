export const emptyBounds = { left: 0, top: 0, width: 0, height: 0 };

export function clampPercent(value) {
  return Math.min(100, Math.max(0, value));
}

export function calculateImageBoundsFromEvent(event) {
  const img = event.detail?.imageEl || event.currentTarget || event.target;
  const naturalWidth = img?.naturalWidth;
  const naturalHeight = img?.naturalHeight;
  const clientWidth = img?.clientWidth;
  const clientHeight = img?.clientHeight;

  if (!naturalWidth || !naturalHeight || !clientWidth || !clientHeight) {
    return emptyBounds;
  }

  const naturalRatio = naturalWidth / naturalHeight;
  const stageRatio = clientWidth / clientHeight;

  if (stageRatio > naturalRatio) {
    const width = (naturalRatio / stageRatio) * 100;
    return { left: (100 - width) / 2, top: 0, width, height: 100 };
  }

  const height = (stageRatio / naturalRatio) * 100;
  return { left: 0, top: (100 - height) / 2, width: 100, height };
}
