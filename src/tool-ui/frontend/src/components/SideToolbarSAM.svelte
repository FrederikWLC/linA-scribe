<script>
  import { emojiUrl } from '../static/js/tool/index.js';

  export let promptControlsEnabled = true;
  export let pointMode = 'box';
  export let points = [];
  export let boxes = [];
  export let setPointMode = () => {};
  export let undoPoint = () => {};
  export let clearPoints = () => {};
</script>

<aside class="side-toolbar" aria-hidden="false">
  <button
    type="button"
    class="tool-panel-button {pointMode === 'box' ? 'active' : ''}"
    on:click={() => setPointMode('box')}
    disabled={!promptControlsEnabled}
    aria-pressed={pointMode === 'box'}
    aria-label="Box"
    title="Box"
  >
    <img src={emojiUrl('🟦')} alt="" class="tool-icon" />
  </button>

  <button
    type="button"
    class="tool-panel-button {pointMode === 'foreground' ? 'active' : ''}"
    on:click={() => setPointMode('foreground')}
    disabled={!promptControlsEnabled}
    aria-pressed={pointMode === 'foreground'}
    aria-label="Foreground point"
    title="Foreground"
  >
    <img src={emojiUrl('🟢')} alt="" class="tool-icon" />
  </button>

  <button
    type="button"
    class="tool-panel-button {pointMode === 'background' ? 'active' : ''}"
    on:click={() => setPointMode('background')}
    disabled={!promptControlsEnabled}
    aria-pressed={pointMode === 'background'}
    aria-label="Background point"
    title="Background"
  >
    <img src={emojiUrl('🔴')} alt="" class="tool-icon" />
  </button>

  <button
    type="button"
    class="tool-panel-button {pointMode === 'delete' ? 'active' : ''}"
    on:click={() => setPointMode('delete')}
    disabled={!promptControlsEnabled}
    aria-pressed={pointMode === 'delete'}
    aria-label="Delete point"
    title="Delete"
  >
    <img src={emojiUrl('🧼')} alt="" class="tool-icon" />
  </button>

  <button
    type="button"
    class="tool-panel-button"
    on:click={undoPoint}
    disabled={!promptControlsEnabled || (points.length === 0 && boxes.length === 0)}
    aria-label="Undo point"
    title="Undo"
  >
    <img src={emojiUrl('↩')} alt="" class="tool-icon" />
  </button>

  <button
    type="button"
    class="tool-panel-button"
    on:click={clearPoints}
    disabled={!promptControlsEnabled || (points.length === 0 && boxes.length === 0)}
    aria-label="Clear points"
    title="Clear"
  >
    <img src={emojiUrl('🗑')} alt="" class="tool-icon" />
  </button>
</aside>