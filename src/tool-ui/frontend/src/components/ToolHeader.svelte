<script>
  export let acceptsPrompts = false;
</script>


<div class="tool-heading">
  <div>
    <p>
      {#if acceptsPrompts}
        Import an image, then click to place foreground or background prompts.
      {:else}
        Import an image, then run to segment.
      {/if}
    </p>
  </div>
</div>
