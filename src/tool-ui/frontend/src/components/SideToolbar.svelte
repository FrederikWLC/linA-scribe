<script>
  export let promptControlsEnabled = true;
  export let pointMode = 'foreground';
  export let points = [];
  export let setPointMode = () => {};
  export let undoPoint = () => {};
  export let clearPoints = () => {};
</script>

<aside class="side-toolbar" aria-hidden="false">
  <button
    type="button"
    class="tool-panel-button {pointMode === 'foreground' ? 'active' : ''}"
    on:click={() => setPointMode('foreground')}
    disabled={!promptControlsEnabled}
    aria-label="Foreground point"
    title="Foreground"
  >
    🟢
  </button>
  <button
    type="button"
    class="tool-panel-button {pointMode === 'background' ? 'active' : ''}"
    on:click={() => setPointMode('background')}
    disabled={!promptControlsEnabled}
    aria-label="Background point"
    title="Background"
  >
    🔴
  </button>
  <button
    type="button"
    class="tool-panel-button {pointMode === 'delete' ? 'active' : ''}"
    on:click={() => setPointMode('delete')}
    disabled={!promptControlsEnabled}
    aria-label="Delete point"
    title="Delete"
  >
    ✖
  </button>
  <button
    type="button"
    class="tool-panel-button"
    on:click={undoPoint}
    disabled={!promptControlsEnabled || points.length === 0}
    aria-label="Undo point"
    title="Undo"
  >
    ↶
  </button>
  <button
    type="button"
    class="tool-panel-button"
    on:click={clearPoints}
    disabled={!promptControlsEnabled || points.length === 0}
    aria-label="Clear points"
    title="Clear"
  >
    🧹
  </button>
</aside>
