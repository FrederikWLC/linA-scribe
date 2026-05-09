<script>
  export let points = [];
  export let imageBounds = { left: 0, top: 0, width: 0, height: 0 };

  const pointSize = 0.8;

  function clampPercent(value) {
    return Math.min(100, Math.max(0, value));
  }

  function imagePointStyle(point) {
    const left = clampPercent(imageBounds.left + (point.x / 100) * imageBounds.width);
    const top = clampPercent(imageBounds.top + (point.y / 100) * imageBounds.height);
    return `left: ${left}%; top: ${top}%; width: ${pointSize}rem; height: ${pointSize}rem;`;
  }

  $: visiblePoints = points.filter(Boolean);
</script>

{#if imageBounds.width > 0 && imageBounds.height > 0}
  {#each visiblePoints as point (point.id)}
    <span
      class:foreground={point.kind === 'foreground'}
      class:background={point.kind === 'background'}
      class="point"
      style={imagePointStyle(point)}
      title={`${point.kind} point`}
    ></span>
  {/each}
{/if}
