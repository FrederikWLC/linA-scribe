<script>
  import { fade } from 'svelte/transition';
  import { onMount } from 'svelte';
  import '../static/css/ToolPage.css';
  import { createToolPageController } from '../static/js/tool/index.js';
  import ToolHeader from '../components/ToolHeader.svelte';
  import ToolBar from '../components/ToolBar.svelte';
  import CanvasRoot from '../components/CanvasRoot.svelte';
  import StatusBar from '../components/StatusBar.svelte';

  export let token = '';
  export let routeModelID = '';
  export let onRouteChange = () => {};

  let fileInput;

  const toolPage = createToolPageController({
    getAuthHeaders: () => (token ? { Authorization: `Bearer ${token}` } : {}),
    onModelSelected: (modelID) => {
      const route =
        modelID === 'sam'
          ? '/tool/sam'
          : modelID === 'classic'
          ? '/tool/classic'
          : '/tool';

      if (window.location.pathname !== route) {
        onRouteChange(route);
      }
    }
  });

  const {
    selectedModel,
    acceptsPrompts,
    imageUrl,
    imageName,
    segmentationImageUrl,
    activeImageMode,
    displayedImageUrl,
    isSegmentationActive,
    imageBounds,
    pointMode,
    points,
    boxes,
    previewBox,
    runMessage,
    importMessage,
    isImageSet,
    isSettingImage,
    canRunPredict,
    foregroundCount,
    backgroundCount,
    selectModel
  } = toolPage;

  $: if (routeModelID && $selectedModel?.id !== routeModelID) {
    selectModel(routeModelID);
  }

  onMount(async () => {
    await toolPage.warmupSAMForUser();
    return toolPage.bindToolPageShortcuts();
  });
</script>

<div class="tool-page">
  <section class="annotator" aria-labelledby="annotator-title">
    <ToolHeader
      acceptsPrompts={$acceptsPrompts}
    />

    <ToolBar
      imageUrl={$imageUrl}
      segmentationImageUrl={$segmentationImageUrl}
      imageName={$imageName}
      acceptsPrompts={$acceptsPrompts}
      activeImageMode={$activeImageMode}
      onImportChange={toolPage.importFromFiles}
      showRawImage={toolPage.showRawImage}
      showSegmentationImage={toolPage.showSegmentationImage}
      runPrompt={toolPage.runPrompt}
      canRunPredict={$canRunPredict}
      canRunExport={$isSegmentationActive}
      isSettingImage={$isSettingImage}
      bind:fileInput={fileInput}
    />

    <CanvasRoot
      sliderControlEnabled={$activeImageMode === 'segmentation'}
      selectedModelID={$selectedModel?.id || routeModelID}
      acceptsPrompts={$acceptsPrompts}
      promptControlsEnabled={$acceptsPrompts && $isImageSet && $activeImageMode === 'raw'}
      pointMode={$pointMode}
      points={$points}
      boxes={$boxes}
      previewBox={$previewBox}
      setPointMode={(mode) => pointMode.set(mode)}
      undoPoint={toolPage.undoPoint}
      clearPoints={toolPage.clearPoints}
      displayedImageUrl={$displayedImageUrl}
      imageName={$imageName}
      isSegmentationActive={$isSegmentationActive}
      activeImageMode={$activeImageMode}
      imageBounds={$imageBounds}
      onPointerDown={toolPage.beginPointSession}
      onPointerMove={toolPage.continuePointSession}
      onPointerUp={toolPage.endPointSession}
      onDrop={toolPage.handleDrop}
      onImageLoad={toolPage.updateImageBounds}
      onClassicGranularityChange={toolPage.runPrompt}
      {fileInput}
    />

    <StatusBar
      acceptsPrompts={$acceptsPrompts}
      foregroundCount={$foregroundCount}
      backgroundCount={$backgroundCount}
      importMessage={$importMessage}
      activeImageMode={$activeImageMode}
      selectedModelLabel={$selectedModel?.label || ''}
    />

    {#if $runMessage}
      <p class="run-message" transition:fade>{$runMessage}</p>
    {/if}
  </section>
</div>
