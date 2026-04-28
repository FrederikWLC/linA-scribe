<script>
  export let boxes = [];
  export let imageBounds = { left: 0, top: 0, width: 0, height: 0 };

  function clampPercent(value) {
    return Math.min(100, Math.max(0, value));
  }

  function boxStyle(box) {
    const left = clampPercent(imageBounds.left + (box.x / 100) * imageBounds.width);
    const top = clampPercent(imageBounds.top + (box.y / 100) * imageBounds.height);
    const width = clampPercent((box.width / 100) * imageBounds.width);
    const height = clampPercent((box.height / 100) * imageBounds.height);
    return `left: ${left}%; top: ${top}%; width: ${width}%; height: ${height}%;`;
  }
</script>

{#if imageBounds.width > 0 && imageBounds.height > 0}
  {#each boxes as box (box.id)}
    <span class="box" style={boxStyle(box)} title="Box prompt"></span>
  {/each}
{/if}
