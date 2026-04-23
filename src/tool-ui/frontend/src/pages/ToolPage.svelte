<script>
  import { onMount } from 'svelte';
  import '../static/css/ToolPage.css';
  import { createToolPageController } from '../static/js/ToolPage.js';

  export let currentUser = '';
  export let token = '';
  export let onLogout = () => {};

  let fileInput;
  const toolPage = createToolPageController({
    getAuthHeaders: () => (token ? { Authorization: `Bearer ${token}` } : {})
  });
  const {
    modelOptions,
    selectedModelKey,
    selectedModel,
    acceptsPrompts,
    imageUrl,
    imageName,
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
    backgroundCount
  } = toolPage;

  onMount(() => {
    toolPage.warmupModels();
    return toolPage.bindToolPageShortcuts();
  });
</script>

<div class="tool-page">
  <header class="page-header">
    <div>
      <h1>Tool</h1>
      <p>Logged in as <strong>{currentUser}</strong></p>
    </div>
    <div class="session-actions">
      <button type="button" on:click={onLogout}>Logout</button>
    </div>
  </header>

  <section class="annotator" aria-labelledby="annotator-title">
    <div class="tool-heading">
      <div>
        <h2 id="annotator-title">Point Prompt Draft</h2>
        <p>
          {#if $acceptsPrompts}
            Import an image, then click to place foreground or background guidance points.
          {:else}
            Import an image, then run {$selectedModel.label} without point prompts.
          {/if}
        </p>
      </div>
      <button type="button" class="run-button" on:click={toolPage.runPrompt} disabled={!$canRunPredict}>
        {$isSettingImage ? 'Setting image...' : 'Run'}
      </button>
    </div>

    <div class="model-picker">
      <label for="model-select">Model</label>
      <select id="model-select" value={$selectedModelKey} on:change={(event) => toolPage.selectModel(event.currentTarget.value)}>
        {#each modelOptions as model}
          <option value={model.key}>{model.label}</option>
        {/each}
      </select>
    </div>

    <div class="toolbar" aria-label="Point controls">
      <input
        bind:this={fileInput}
        class="file-input"
        type="file"
        accept="image/*"
        on:change={toolPage.importFromFiles}
      />
      <button type="button" on:click={() => toolPage.openFilePicker(fileInput)}>Import image</button>
      <button
        type="button"
        class:active={$activeImageMode === 'raw'}
        on:click={toolPage.showRawImage}
        disabled={!$imageUrl}
      >
        Raw image
      </button>
      <button
        type="button"
        class:active={$activeImageMode === 'segmentation'}
        on:click={toolPage.showSegmentationImage}
        disabled={!$segmentationImageUrl}
      >
        Segmentation
      </button>
      <button
        type="button"
        class:active={$pointMode === 'foreground'}
        on:click={() => pointMode.set('foreground')}
        disabled={!$acceptsPrompts}
      >
        Green foreground
      </button>
      <button
        type="button"
        class:active={$pointMode === 'background'}
        on:click={() => pointMode.set('background')}
        disabled={!$acceptsPrompts}
      >
        Red background
      </button>
      <button type="button" on:click={toolPage.undoPoint} disabled={!$acceptsPrompts || $points.length === 0}>Undo</button>
      <button type="button" on:click={toolPage.clearPoints} disabled={!$acceptsPrompts || $points.length === 0}>Clear</button>
    </div>

    {#if $displayedImageUrl}
      <button
        type="button"
        class="image-stage"
        class:segmentation-active={$isSegmentationActive}
        on:click={toolPage.placePoint}
        on:drop={toolPage.handleDrop}
        on:dragover={toolPage.keepDropActive}
        aria-label={$acceptsPrompts ? 'Place point on image' : 'Preview image'}
      >
        <img
          class:segmentation-image={$isSegmentationActive}
          src={$displayedImageUrl}
          alt={$imageName || 'Imported point prompting target'}
          on:load={toolPage.updateImageBounds}
        />
        {#if $activeImageMode === 'raw' && $acceptsPrompts}
          {#each $points as point, index (point.id)}
            <span
              class:foreground={point.kind === 'foreground'}
              class:background={point.kind === 'background'}
              class="point"
              style={toolPage.imagePointStyle(point, $imageBounds)}
              title={`${point.kind} point ${index + 1}`}
            >
              {index + 1}
            </span>
          {/each}
        {/if}
      </button>
    {:else}
      <button
        type="button"
        class="drop-zone"
        on:drop={toolPage.handleDrop}
        on:dragover={toolPage.keepDropActive}
        on:click={() => toolPage.openFilePicker(fileInput)}
      >
        <strong>Drop an image here</strong>
        <span>Paste with Ctrl/Cmd+V or import from files.</span>
      </button>
    {/if}

    <div class="status-row">
      <p>
        {#if $acceptsPrompts}
          <strong>{$foregroundCount}</strong> foreground / <strong>{$backgroundCount}</strong> background
        {:else}
          Prompts disabled for {$selectedModel.label}
        {/if}
      </p>
      <p>
        {$importMessage} View: {$activeImageMode === 'segmentation' ? 'segmentation' : 'raw image'}. Undo shortcut:
        Ctrl/Cmd+Z
      </p>
    </div>

    {#if $runMessage}
      <p class="run-message">{$runMessage}</p>
    {/if}
  </section>
</div>
