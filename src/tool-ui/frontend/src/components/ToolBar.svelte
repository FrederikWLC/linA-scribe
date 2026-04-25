<script>
  export let imageUrl = '';
  export let segmentationImageUrl = '';
  export let acceptsPrompts = false;
  export let promptControlsEnabled = true;
  export let pointMode = 'foreground';
  export let points = [];
  export let activeImageMode = 'raw';
  export let onImportChange = () => {};
  export let showRawImage = () => {};
  export let showSegmentationImage = () => {};
  export let setPointMode = () => {};
  export let undoPoint = () => {};
  export let clearPoints = () => {};
  export let fileInput;
</script>

<div class="toolbar" aria-label={acceptsPrompts ? 'Point controls' : 'Image controls'}>
  <input
    bind:this={fileInput}
    class="file-input"
    type="file"
    accept="image/*"
    on:change={onImportChange}
  />
  <button type="button" on:click={() => fileInput?.click()}>Import image</button>
  <button
    type="button"
    class:active={activeImageMode === 'raw'}
    on:click={showRawImage}
    disabled={!imageUrl}
  >
    Raw image
  </button>
  <button
    type="button"
    class:active={activeImageMode === 'segmentation'}
    on:click={showSegmentationImage}
    disabled={!segmentationImageUrl}
  >
    Segmentation
  </button>

  {#if acceptsPrompts}
    <button
      type="button"
      class:active={pointMode === 'foreground'}
      on:click={() => setPointMode('foreground')}
      disabled={!promptControlsEnabled}
    >
      Green
    </button>
    <button
      type="button"
      class:active={pointMode === 'background'}
      on:click={() => setPointMode('background')}
      disabled={!promptControlsEnabled}
    >
      Red
    </button>
    <button
      type="button"
      class:active={pointMode === 'delete'}
      on:click={() => setPointMode('delete')}
      disabled={!promptControlsEnabled}
    >
      Delete
    </button>
    <button type="button" on:click={undoPoint} disabled={!promptControlsEnabled || points.length === 0}>Undo</button>
    <button type="button" on:click={clearPoints} disabled={!promptControlsEnabled || points.length === 0}>Clear</button>
  {/if}
</div>
