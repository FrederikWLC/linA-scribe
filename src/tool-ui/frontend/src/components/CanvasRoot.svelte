<script>
  import ImageLayer from './ImageLayer.svelte';
  import PointsLayer from './PointsLayer.svelte';
  import BoxesLayer from './BoxesLayer.svelte';
  import SideToolbar from './SideToolbar.svelte';

  export let displayedImageUrl = '';
  export let imageName = '';
  export let isSegmentationActive = false;
  export let activeImageMode = 'raw';
  export let acceptsPrompts = false;
  export let promptControlsEnabled = true;
  export let pointMode = 'foreground';
  export let points = [];
  export let boxes = [];
  export let previewBox = null;
  export let imageBounds = { left: 0, top: 0, width: 0, height: 0 };
  export let setPointMode = () => {};
  export let undoPoint = () => {};
  export let clearPoints = () => {};
  export let onPointerDown = () => {};
  export let onPointerMove = () => {};
  export let onPointerUp = () => {};
  export let onDrop = () => {};
  export let onImageLoad = () => {};
  export let fileInput;

  function clampPercent(value) {
    return Math.min(100, Math.max(0, value));
  }

  function previewBoxStyle(box) {
    const left = clampPercent(imageBounds.left + (box.x1 / 100) * imageBounds.width);
    const top = clampPercent(imageBounds.top + (box.y1 / 100) * imageBounds.height);
    const width = clampPercent(((box.x2 - box.x1) / 100) * imageBounds.width);
    const height = clampPercent(((box.y2 - box.y1) / 100) * imageBounds.height);
    return `left: ${left}%; top: ${top}%; width: ${width}%; height: ${height}%;`;
  }
</script>

<div class="canvas-root">
  <SideToolbar
    promptControlsEnabled={promptControlsEnabled}
    pointMode={pointMode}
    points={points}
    boxes={boxes}
    setPointMode={setPointMode}
    undoPoint={undoPoint}
    clearPoints={clearPoints}
  />
  <div class="canvas-frame">
    {#if displayedImageUrl}
      <button
        type="button"
        class="image-stage"
        class:segmentation-active={isSegmentationActive}
        on:pointerdown|preventDefault={onPointerDown}
        on:pointermove|preventDefault={onPointerMove}
        on:pointerup|preventDefault={onPointerUp}
        on:pointercancel|preventDefault={onPointerUp}
        on:drop={onDrop}
        on:dragover|preventDefault
        aria-label={acceptsPrompts ? 'Place point on image' : 'Preview image'}
      >
        <ImageLayer
          {displayedImageUrl}
          {imageName}
          {isSegmentationActive}
          on:load={onImageLoad}
        />

        {#if activeImageMode === 'raw' && acceptsPrompts && imageBounds.width > 0 && imageBounds.height > 0}
          <BoxesLayer {boxes} {imageBounds} />
          {#if previewBox}
            <span class="box preview" style={previewBoxStyle(previewBox)} title="Preview box"></span>
          {/if}
          <PointsLayer {points} {imageBounds} />
        {/if}
      </button>
    {:else}
      <button
        type="button"
        class="drop-zone"
        on:drop={onDrop}
        on:dragover|preventDefault
        on:click={() => fileInput?.click()}
      >
        <strong>Drop an image here</strong>
        <span>Paste with Ctrl/Cmd+V or import from files.</span>
      </button>
    {/if}
  </div>
</div>
