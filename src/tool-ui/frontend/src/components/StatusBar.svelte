<script>
  import { classicGranularity } from '../static/js/tool/classicGranularity.js';
  export let acceptsPrompts = false;
  export let foregroundCount = 0;
  export let backgroundCount = 0;
  export let importMessage = '';
  export let activeImageMode = 'raw';
  export let selectedModelLabel = '';
</script>

<div class="status-row">
  <p>
    {#if acceptsPrompts}
      <strong>{foregroundCount}</strong> foreground / <strong>{backgroundCount}</strong> background
    {:else}
      Granularity: {#if $classicGranularity === 0}Default{:else}<strong>{#if $classicGranularity > 0}+{/if}{$classicGranularity}</strong> / 100{/if}
    {/if}
  </p>
  <p>
    {importMessage} View: {activeImageMode === 'segmentation' ? 'segmentation' : 'raw image'}. Undo shortcut:
    Ctrl/Cmd+Z
  </p>
</div>
