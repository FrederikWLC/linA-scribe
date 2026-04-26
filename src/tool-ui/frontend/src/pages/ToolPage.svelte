<script>
  import { onMount } from 'svelte';
  import '../static/css/ToolPage.css';
  import { createToolPageController } from '../static/js/tool/index.js';
  import ToolHeader from '../components/ToolHeader.svelte';
  import ToolBar from '../components/ToolBar.svelte';
  import CanvasRoot from '../components/CanvasRoot.svelte';
  import StatusBar from '../components/StatusBar.svelte';

  export let currentUser = '';
  export let token = '';
  export let onLogout = () => {};
  export let initialModelKey = '';
  export let onRouteChange = () => {};

  let fileInput;

  const toolPage = createToolPageController({
    getAuthHeaders: () => (token ? { Authorization: `Bearer ${token}` } : {}),
    initialModelKey,
    onModelSelected: (modelKey) => {
      const route =
        modelKey === 'modal-mobilesam'
          ? '/tool/sam'
          : modelKey === 'gaussian'
          ? '/tool/gaussian'
          : modelKey === 'grabcut-auto-brush'
          ? '/tool/grabcut'
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
    runMessage,
    importMessage,
    isImageSet,
    isSettingImage,
    canRunPredict,
    foregroundCount,
    backgroundCount
  } = toolPage;

  onMount(() => {
    toolPage.warmupModels();
    return toolPage.bindToolPageShortcuts();
  });
</script>

<div class="tool-page">
  <section class="annotator" aria-labelledby="annotator-title">
    <ToolHeader
      currentUser={currentUser}
      selectedModelLabel={$selectedModel.label}
      acceptsPrompts={$acceptsPrompts}
      isSettingImage={$isSettingImage}
      canRunPredict={$canRunPredict}
      canRunExport={$isSegmentationActive}
      segmentationImageUrl={$segmentationImageUrl}
      imageName={$imageName}
      runPrompt={toolPage.runPrompt}
      onLogout={onLogout}
    />

    <ToolBar
      imageUrl={$imageUrl}
      segmentationImageUrl={$segmentationImageUrl}
      acceptsPrompts={$acceptsPrompts}
      promptControlsEnabled={$acceptsPrompts && (!$selectedModel.requiresSetImage || $isImageSet)}
      activeImageMode={$activeImageMode}
      pointMode={$pointMode}
      points={$points}
      onImportChange={toolPage.importFromFiles}
      showRawImage={toolPage.showRawImage}
      showSegmentationImage={toolPage.showSegmentationImage}
      setPointMode={(mode) => pointMode.set(mode)}
      undoPoint={toolPage.undoPoint}
      clearPoints={toolPage.clearPoints}
      bind:fileInput={fileInput}
    />

    <CanvasRoot
      displayedImageUrl={$displayedImageUrl}
      imageName={$imageName}
      isSegmentationActive={$isSegmentationActive}
      activeImageMode={$activeImageMode}
      acceptsPrompts={$acceptsPrompts}
      points={$points}
      imageBounds={$imageBounds}
      onPointerDown={toolPage.beginPointSession}
      onPointerMove={toolPage.continuePointSession}
      onPointerUp={toolPage.endPointSession}
      onDrop={toolPage.handleDrop}
      onImageLoad={toolPage.updateImageBounds}
      {fileInput}
    />

    <StatusBar
      acceptsPrompts={$acceptsPrompts}
      foregroundCount={$foregroundCount}
      backgroundCount={$backgroundCount}
      importMessage={$importMessage}
      activeImageMode={$activeImageMode}
      selectedModelLabel={$selectedModel.label}
    />

    {#if $runMessage}
      <p class="run-message">{$runMessage}</p>
    {/if}
  </section>
</div>
