<script>
  import { exportImage } from '../static/js/tool/export.js';

  export let imageUrl = '';
  export let segmentationImageUrl = '';
  export let imageName = '';
  export let acceptsPrompts = false;
  export let activeImageMode = 'raw';
  export let onImportChange = () => {};
  export let showRawImage = () => {};
  export let showSegmentationImage = () => {};
  export let runPrompt = () => {};
  export let canRunPredict = false;
  export let canRunExport = false;
  export let isSettingImage = false;
  export let isDecodingMask = false;
  export let fileInput;

  let isExportMenuOpen = false;

  function toggleExportMenu() {
    if (!canRunExport) {
      return;
    }
    isExportMenuOpen = !isExportMenuOpen;
  }

  function closeExportMenu() {
    isExportMenuOpen = false;
  }
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
    Raw
  </button>
  <button
    type="button"
    class:active={activeImageMode === 'segmentation'}
    on:click={showSegmentationImage}
    disabled={!segmentationImageUrl}
  >
    Mask
  </button>

  <span class="toolbar-spacer"></span>

  <button
    type="button"
    class="run-button"
    class:loading={isDecodingMask}
    on:click={runPrompt}
    disabled={!canRunPredict}
    aria-label={isDecodingMask ? 'Decoding mask' : 'Run'}
    aria-busy={isDecodingMask}
  >
    {#if isDecodingMask}
      <span class="run-loader" aria-hidden="true"></span>
    {:else}
      {isSettingImage ? 'Setting image...' : 'Run'}
    {/if}
  </button>
  <div class="export-dropdown">
    <button
      type="button"
      class="export-summary"
      on:click={toggleExportMenu}
      disabled={!canRunExport}
      aria-haspopup="menu"
      aria-expanded={isExportMenuOpen}
    >
      Export
    </button>
    {#if isExportMenuOpen}
      <ul class="export-options" role="menu">
        <li><button type="button" role="menuitem" on:click={() => exportImage(segmentationImageUrl, '.png', imageName || 'segmentation').catch((error) => console.error(error)).finally(closeExportMenu)}>.png</button></li>
        <li><button type="button" role="menuitem" on:click={() => exportImage(segmentationImageUrl, '.jpg', imageName || 'segmentation').catch((error) => console.error(error)).finally(closeExportMenu)}>.jpg</button></li>
        <li><button type="button" role="menuitem" on:click={() => exportImage(segmentationImageUrl, '.ora', imageName || 'segmentation', imageUrl).catch((error) => console.error(error)).finally(closeExportMenu)}>.ora</button></li>
      </ul>
    {/if}
  </div>
</div>
