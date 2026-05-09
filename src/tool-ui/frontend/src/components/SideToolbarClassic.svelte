<script>
  import { emojiUrl } from '../static/js/tool/index.js';
  import { classicGranularity } from '../static/js/tool/classicGranularity.js';

  export let onGranularityChange = () => {};
  export let sliderControlEnabled = true;

  function resetGranularity() {
    classicGranularity.set(0);
    onGranularityChange();
  }
</script>
<style>
  .classic-slider {
    height:100%;
    max-height:550px;
  }
  
</style>
<aside class="side-toolbar classic-toolbar" aria-hidden="false">
  <button
    type="button"
    class="tool-panel-button"
    onclick={() => { resetGranularity(); }}
    disabled={!sliderControlEnabled}
  >
    <img src={emojiUrl('✨')} alt="" class="tool-icon" />
  </button>
  <input
    id="classic-granularity"
    class="classic-slider"
    type="range"
    min="-50"
    max="100"
    step="1"
    bind:value={$classicGranularity}
    style={`--value: ${$classicGranularity};`}
    onchange={onGranularityChange}
    orient="vertical"
    aria-label="Classic granularity"
    disabled={!sliderControlEnabled}
  />
</aside>
