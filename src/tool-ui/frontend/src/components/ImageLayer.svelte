<script>
  import { createEventDispatcher, onMount, onDestroy } from 'svelte';

  const dispatch = createEventDispatcher();
  let imageEl;

  export let displayedImageUrl = '';
  export let imageName = '';
  export let isSegmentationActive = false;

  function handleLoad(event) {
    dispatch('load', {
      imageEl: event.currentTarget || event.target
    });
  }

  function notifySizeChange() {
    if (imageEl) {
      dispatch('load', { imageEl });
    }
  }

  onMount(() => {
    window.addEventListener('resize', notifySizeChange);
  });

  onDestroy(() => {
    window.removeEventListener('resize', notifySizeChange);
  });
</script>

<img
  bind:this={imageEl}
  class:segmentation-image={isSegmentationActive}
  src={displayedImageUrl}
  alt={imageName || 'Imported point prompting target'}
  on:load={handleLoad}
/>
