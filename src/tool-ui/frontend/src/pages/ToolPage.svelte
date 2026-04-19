<script>
  import { onMount } from 'svelte';
  import '../static/css/ToolPage.css';
  import { createToolPageController } from '../static/js/ToolPage.js';

  export let currentUser = '';
  export let protectedMessage = '';
  export let onReloadProtected = () => {};
  export let onLogout = () => {};

  let fileInput;
  const toolPage = createToolPageController();
  const {
    imageUrl,
    imageName,
    imageBounds,
    pointMode,
    points,
    runMessage,
    importMessage,
    foregroundCount,
    backgroundCount
  } = toolPage;

  onMount(() => {
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
      <button type="button" on:click={onReloadProtected}>Reload Protected Data</button>
      <button type="button" on:click={onLogout}>Logout</button>
    </div>
  </header>

  <section class="annotator" aria-labelledby="annotator-title">
    <div class="tool-heading">
      <div>
        <h2 id="annotator-title">Point Prompt Draft</h2>
        <p>Import an image, then click to place foreground or background guidance points.</p>
      </div>
      <button type="button" class="run-button" on:click={toolPage.runPrompt} disabled={!$imageUrl}>Run</button>
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
        class:active={$pointMode === 'foreground'}
        on:click={() => pointMode.set('foreground')}
      >
        Green foreground
      </button>
      <button
        type="button"
        class:active={$pointMode === 'background'}
        on:click={() => pointMode.set('background')}
      >
        Red background
      </button>
      <button type="button" on:click={toolPage.undoPoint} disabled={$points.length === 0}>Undo</button>
      <button type="button" on:click={toolPage.clearPoints} disabled={$points.length === 0}>Clear</button>
    </div>

    {#if $imageUrl}
      <button
        type="button"
        class="image-stage"
        on:click={toolPage.placePoint}
        on:drop={toolPage.handleDrop}
        on:dragover={toolPage.keepDropActive}
        aria-label="Place point on image"
      >
        <img
          src={$imageUrl}
          alt={$imageName || 'Imported point prompting target'}
          on:load={toolPage.updateImageBounds}
        />
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
      <p><strong>{$foregroundCount}</strong> foreground / <strong>{$backgroundCount}</strong> background</p>
      <p>{$importMessage} Undo shortcut: Ctrl/Cmd+Z</p>
    </div>

    {#if $runMessage}
      <p class="run-message">{$runMessage}</p>
    {/if}
  </section>

  <section class="protected-page">
    <h2>Protected Page</h2>
    <p>{protectedMessage}</p>
  </section>
</div>
