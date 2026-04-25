<script>
  import ImageLayer from './ImageLayer.svelte';
  import PointsLayer from './PointsLayer.svelte';

  export let displayedImageUrl = '';
  export let imageName = '';
  export let isSegmentationActive = false;
  export let activeImageMode = 'raw';
  export let acceptsPrompts = false;
  export let points = [];
  export let imageBounds = { left: 0, top: 0, width: 0, height: 0 };
  export let onPointerDown = () => {};
  export let onPointerMove = () => {};
  export let onPointerUp = () => {};
  export let onDrop = () => {};
  export let onImageLoad = () => {};
  export let fileInput;
</script>

{#if displayedImageUrl}
  <div class="canvas-root">
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
        <PointsLayer {points} {imageBounds} />
      {/if}
    </button>

  </div>
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
