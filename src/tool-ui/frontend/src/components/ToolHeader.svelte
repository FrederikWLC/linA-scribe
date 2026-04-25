<script>
  export let currentUser = '';
  export let selectedModelLabel = '';
  export let acceptsPrompts = false;
  export let isSettingImage = false;
  export let canRunPredict = false;
  export let runPrompt = () => {};
  export let onLogout = () => {};
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
  <button type="button" class="run-button" on:click={runPrompt} disabled={!canRunPredict}>
    {isSettingImage ? 'Setting image...' : 'Run'}
  </button>
</div>
