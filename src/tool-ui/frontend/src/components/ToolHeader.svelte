<script>
  import { exportImage } from '../static/js/tool/export.js';

  export let currentUser = '';
  export let selectedModelLabel = '';
  export let acceptsPrompts = false;
  export let isSettingImage = false;
  export let canRunPredict = false;
  export let canRunExport = false;
  export let segmentationImageUrl = '';
  export let imageName = '';
  export let runPrompt = () => {};
  export let onLogout = () => {};

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

<header class="page-header">
  <div>
    <h1>Tool</h1>
    <p>Logged in as <strong>{currentUser}</strong></p>
  </div>
  <div class="session-actions">
    <button type="button" on:click={onLogout}>Logout</button>
  </div>
</header>

<div class="tool-heading">
  <div>
    <h2 id="annotator-title">Point Prompt Draft</h2>
    <p>
      {#if acceptsPrompts}
        Import an image, then click to place foreground or background guidance points.
      {:else}
        Import an image, then run {selectedModelLabel} without point prompts.
      {/if}
    </p>
  </div>
  <div class="action-buttons">
    <button type="button" class="run-button" on:click={runPrompt} disabled={!canRunPredict}>
      {isSettingImage ? 'Setting image...' : 'Run'}
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
          <li><button type="button" role="menuitem" on:click={() => exportImage(segmentationImageUrl, '.ora', imageName || 'segmentation').catch((error) => console.error(error)).finally(closeExportMenu)}>.ora</button></li>
        </ul>
      {/if}
    </div>
  </div>
</div>
